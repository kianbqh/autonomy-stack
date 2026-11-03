"""Tiny synthetic ZIP duplicate contracts; no real archives, SLAM or network."""
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest import mock
import warnings
import zipfile
import zlib

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'python'))
import import_kitti as importer

FULL='data_odometry_calib.zip'
COLOR='data_odometry_color.zip'
POSES='data_odometry_poses.zip'
SCANS='data_odometry_velodyne.zip'

def calibration(full=False,changed=None):
    rows={key:['1','0','2','0','0','1','3','0','0','0','1','0'] for key in ('P0','P1','P2','P3')}
    if full:rows['Tr']=['1','0','0','0','0','1','0','0','0','0','1','0']
    if changed: changed(rows)
    return ''.join(key+': '+' '.join(values)+'\n' for key,values in rows.items()).encode('ascii')

class DuplicateImportContracts(unittest.TestCase):
    def setUp(self):
        scratch=Path(os.environ.get('M3_IMPORT_TEST_TMP',ROOT/'results/.import-test-tmp')).resolve()
        scratch.mkdir(parents=True,exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(prefix='synthetic-import-',dir=scratch)
        self.root=Path(self.temp.name).resolve()
        self.assertTrue(self.root.is_relative_to(scratch))
        self.addCleanup(self.temp.cleanup)
        self.counter=0

    def archive(self,label,members):
        self.counter+=1
        folder=self.root/f'archive-{self.counter}'
        folder.mkdir()
        path=folder/label
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',UserWarning)
            with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_STORED) as z:
                for name,payload in members:z.writestr(name,payload)
        return {'path':str(path),'bytes':path.stat().st_size,'sha256':importer.sha256(path)}

    def state(self):
        self.counter+=1
        out=self.root/f'extracted-{self.counter}'
        out.mkdir()
        return out,{},{},{},{}

    def consume(self,receipt,name,state,sequence='07'):
        out,hashes,crc,sources,canonical=state
        with zipfile.ZipFile(receipt['path']) as z:
            importer.import_selected(z,z.getinfo(name),receipt,out,name,sequence,hashes,crc,sources,canonical)

    def test_byte_identical_ordinary_duplicate_records_both_sources(self):
        name='dataset/sequences/07/times.txt';payload=b'0.0\n0.1\n'
        a=self.archive(FULL,[(name,payload)]);b=self.archive(COLOR,[(name,payload)])
        state=self.state()
        self.consume(a,name,state);self.consume(b,name,state)
        out,hashes,crc,sources,canonical=state
        self.assertEqual((out/name).read_bytes(),payload)
        self.assertEqual(hashes[name],hashlib.sha256(payload).hexdigest())
        self.assertEqual(crc[name],f'{zlib.crc32(payload):08x}')
        self.assertEqual([s['canonical'] for s in sources[name]],[True,False])
        for source in sources[name]:
            self.assertTrue(source['accepted'] and source['crc_verified'])
            self.assertEqual(source['sha256'],hashes[name])
            self.assertEqual(source['member'],name)
        self.assertEqual(canonical[name]['archive'],FULL)

    def test_ordinary_conflicting_bytes_rejected_and_receipt_retained(self):
        name='dataset/sequences/07/times.txt';a=self.archive(FULL,[(name,b'0\n')]);b=self.archive(COLOR,[(name,b'1\n')])
        state=self.state();self.consume(a,name,state)
        with self.assertRaisesRegex(ValueError,'conflicting selected bytes'):
            self.consume(b,name,state)
        self.assertEqual((state[0]/name).read_bytes(),b'0\n')
        rejected=state[3][name][-1]
        self.assertFalse(rejected['accepted']);self.assertFalse(rejected['canonical'])
        self.assertEqual(rejected['sha256'],hashlib.sha256(b'1\n').hexdigest())
        self.assertTrue(rejected['crc_verified'])

    def test_canonical_full_calibration_in_both_orders_and_configured_sequences(self):
        for sequence in ('07','00'):
            for order in ((FULL,COLOR),(COLOR,FULL)):
                with self.subTest(sequence=sequence,order=order):
                    name=f'dataset/sequences/{sequence}/calib.txt'
                    full=calibration(True);camera=calibration(False)
                    receipts={FULL:self.archive(FULL,[(name,full)]),COLOR:self.archive(COLOR,[(name,camera)])}
                    state=self.state()
                    for label in order:self.consume(receipts[label],name,state,sequence)
                    out,hashes,crc,sources,canonical=state
                    self.assertEqual((out/name).read_bytes(),full)
                    self.assertEqual(hashes[name],hashlib.sha256(full).hexdigest())
                    self.assertEqual(crc[name],f'{zlib.crc32(full):08x}')
                    self.assertEqual(canonical[name]['archive'],FULL)
                    self.assertEqual(sum(source['canonical'] for source in sources[name]),1)
                    self.assertEqual({source['archive'] for source in sources[name]},{FULL,COLOR})
                    self.assertEqual({source['sha256'] for source in sources[name]},
                                     {hashlib.sha256(full).hexdigest(),hashlib.sha256(camera).hexdigest()})
                    self.assertIn(b'Tr:',(out/name).read_bytes())

    def test_exact_numeric_formatting_equivalence_keeps_full_original_bytes(self):
        name='dataset/sequences/07/calib.txt';full=calibration(True)
        camera=calibration().replace(b'1 ',b'1.000e0 ')
        state=self.state()
        for label,payload in ((COLOR,camera),(FULL,full)):
            self.consume(self.archive(label,[(name,payload)]),name,state)
        self.assertEqual((state[0]/name).read_bytes(),full)

    def test_sub_float_precision_common_numeric_difference_is_rejected(self):
        name='dataset/sequences/07/calib.txt'
        changed=calibration(changed=lambda rows:rows['P2'].__setitem__(0,'1.000000000000000000001'))
        for order in ((FULL,COLOR),(COLOR,FULL)):
            with self.subTest(order=order):
                payloads={FULL:calibration(True),COLOR:changed};state=self.state()
                self.consume(self.archive(order[0],[(name,payloads[order[0]])]),name,state)
                with self.assertRaisesRegex(ValueError,'conflicting calibration camera values'):
                    self.consume(self.archive(order[1],[(name,payloads[order[1]])]),name,state)

    def test_missing_camera_field_or_full_transform_rejected(self):
        name='dataset/sequences/07/calib.txt'
        for label,full,removed in [(FULL,True,'Tr'),(FULL,True,'P1'),(COLOR,False,'P3')]:
            with self.subTest(label=label,removed=removed):
                payload=calibration(full,lambda rows:rows.pop(removed))
                with self.assertRaisesRegex(ValueError,'missing calibration fields'):
                    self.consume(self.archive(label,[(name,payload)]),name,self.state())

    def test_invalid_nonfinite_and_overflow_numeric_values_rejected(self):
        name='dataset/sequences/07/calib.txt'
        for token in ('NaN','inf','-inf','1e999','not-a-number','1_0'):
            with self.subTest(token=token):
                payload=calibration(changed=lambda rows:rows['P0'].__setitem__(0,token))
                with self.assertRaises(ValueError):
                    self.consume(self.archive(COLOR,[(name,payload)]),name,self.state())

    def test_extra_or_duplicate_fields_and_wrong_matrix_length_rejected(self):
        name='dataset/sequences/07/calib.txt'
        invalid=[calibration()+b'Tr: '+' '.join(['0']*12).encode()+b'\n',
                 calibration()+calibration().splitlines(keepends=True)[0],
                 calibration(changed=lambda rows:rows['P1'].pop())]
        for payload in invalid:
            with self.subTest(payload=payload):
                with self.assertRaises(ValueError):
                    self.consume(self.archive(COLOR,[(name,payload)]),name,self.state())

    def test_other_archive_source_cannot_use_calibration_subset_exception(self):
        name='dataset/sequences/07/calib.txt';state=self.state()
        self.consume(self.archive(FULL,[(name,calibration(True))]),name,state)
        with self.assertRaisesRegex(ValueError,'conflicting selected bytes'):
            self.consume(self.archive(POSES,[(name,calibration())]),name,state)

    def test_exception_does_not_match_other_path_or_unconfigured_sequence(self):
        for name in ('dataset/sequences/00/calib.txt','dataset/sequences/07/not-calib.txt'):
            with self.subTest(name=name):
                state=self.state()
                self.consume(self.archive(FULL,[(name,calibration(True))]),name,state,sequence='07')
                with self.assertRaisesRegex(ValueError,'conflicting selected bytes'):
                    self.consume(self.archive(COLOR,[(name,calibration())]),name,state,sequence='07')

    def test_corrupt_duplicate_payload_still_fails_zip_crc(self):
        name='dataset/sequences/07/times.txt';payload=b'0.0\n'
        good=self.archive(FULL,[(name,payload)]);bad=self.archive(COLOR,[(name,payload)])
        path=Path(bad['path'])
        with zipfile.ZipFile(path) as z:
            info=z.getinfo(name)
            offset=info.header_offset+30+len(info.filename.encode())+len(info.extra)
        data=bytearray(path.read_bytes());data[offset]^=1;path.write_bytes(data)
        bad.update(bytes=path.stat().st_size,sha256=importer.sha256(path))
        state=self.state();self.consume(good,name,state)
        with self.assertRaises(zipfile.BadZipFile):self.consume(bad,name,state)

    def inputs(self,full_dataset=True,calib_members=None):
        base='dataset/sequences/07/'
        times=('\n'.join(str(i/10) for i in range(1101))+'\n').encode()
        full=[(base+'calib.txt',calibration(True)),(base+'times.txt',times)] if calib_members is None else calib_members
        members={FULL:full, COLOR:[(base+'calib.txt',calibration()),(base+'times.txt',times)],
                 POSES:[('dataset/poses/07.txt',(b'1 0 0 0 0 1 0 0 0 0 1 0\n')*1101)],SCANS:[]}
        if full_dataset:
            members[SCANS]=[(base+f'velodyne/{i:06d}.bin',struct.pack('<4f',float(i),0,0,1)) for i in range(1101)]
            members[COLOR]+=[(base+f'image_2/{i:06d}.png',b'synthetic PNG placeholder') for i in (0,550,1100)]
        receipts={name:self.archive(name,entries) for name,entries in members.items()}
        config=self.root/'config.json'
        config.write_text(json.dumps({'kind':'kitti_full','sequence':'07','expected_frames':1101,'calibration':{'frames':[0,550,1100]}}))
        return receipts,config

    def invoke(self,receipts,config,order,output):
        self.counter+=1
        manifest=self.root/f'archives-{self.counter}.json'
        manifest.write_text(json.dumps({'archives':[receipts[name] for name in order]}))
        argv=['import_kitti.py','--archives',str(manifest),'--config',str(config),'--output',str(output)]
        with mock.patch.object(sys,'argv',argv),contextlib.redirect_stdout(io.StringIO()):
            importer.main()

    def test_full_synthetic_import_both_orders_and_fullhash_still_runs_on_resume(self):
        receipts,config=self.inputs()
        real_sha=importer.sha256
        for order in ((FULL,POSES,SCANS,COLOR),(COLOR,SCANS,POSES,FULL)):
            with self.subTest(order=order):
                self.counter+=1;output=self.root/f'full-output-{self.counter}'
                self.invoke(receipts,config,order,output)
                result=json.loads((output/'input-manifest.json').read_text())
                name='dataset/sequences/07/calib.txt'
                self.assertEqual(len(result['files_sha256']),1107)
                self.assertEqual(result['frame_ids'],list(range(1101)))
                self.assertTrue(result['verified'])
                self.assertEqual(result['canonical_selection'][name]['archive'],FULL)
                self.assertEqual((output/name).read_bytes(),calibration(True))
                self.assertEqual(len(result['member_sources'][name]),2)
                self.assertEqual(len(result['member_sources']['dataset/sequences/07/times.txt']),2)
                with mock.patch.object(importer,'sha256',wraps=real_sha) as spy:
                    self.invoke(receipts,config,order,output)
                archive_paths={Path(entry['path']) for entry in receipts.values()}
                measured=[Path(call.args[0]) for call in spy.call_args_list if Path(call.args[0]) in archive_paths]
                self.assertEqual(measured,[Path(receipts[name]['path']) for name in order])

    def test_color_only_calibration_cannot_be_final_canonical(self):
        receipts,config=self.inputs(False,calib_members=[])
        with self.assertRaisesRegex(ValueError,'complete official calibration member required'):
            self.invoke(receipts,config,(FULL,COLOR,POSES,SCANS),self.root/'missing-full')

    def test_full_archive_checksum_mismatch_stops_before_output(self):
        receipts,config=self.inputs(False)
        receipts[SCANS]['sha256']='0'*64;output=self.root/'checksum-rejected'
        with self.assertRaisesRegex(ValueError,'archive checksum mismatch'):
            self.invoke(receipts,config,(FULL,COLOR,POSES,SCANS),output)
        self.assertFalse(output.exists())

    def test_duplicate_within_archive_still_rejected_even_if_unselected(self):
        receipts,config=self.inputs(False,calib_members=[('ignored.txt',b'a'),('ignored.txt',b'a')])
        with self.assertRaisesRegex(ValueError,'duplicate ZIP member'):
            self.invoke(receipts,config,(FULL,COLOR,POSES,SCANS),self.root/'within-archive')

    def test_unsafe_unselected_member_still_rejected(self):
        receipts,config=self.inputs(False,calib_members=[('../outside',b'a')])
        with self.assertRaisesRegex(ValueError,'unsafe ZIP member'):
            self.invoke(receipts,config,(FULL,COLOR,POSES,SCANS),self.root/'unsafe')


if __name__=='__main__':unittest.main()
