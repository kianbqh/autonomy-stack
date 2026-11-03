"""Standard-library fixture checks only; these do not test Fetch or physics."""
import unittest
from manip.sensor_checks_spec import (COMMANDS,FIFO_COMMANDS,FIFO_EXPECTED,ZERO_ACTION,
    NOISE_OFFSETS_M,DROPOUT_DRAWS,DROPOUT_HOLD,DROPOUT_PROBABILITY,HOLD_BLOCKS,SENSOR_COMMANDS)

class SensorFixtureTests(unittest.TestCase):
    def test_fifo_fixture_is_distinguishable_and_drains_all_commands(self):
        self.assertEqual(len(set(COMMANDS)),8)
        self.assertNotIn(ZERO_ACTION,COMMANDS)
        self.assertEqual(len(FIFO_COMMANDS),10)
        for delay in (1,2):
            outputs=FIFO_EXPECTED[delay]
            self.assertEqual(len(outputs),10)
            self.assertEqual(outputs[:delay],(ZERO_ACTION,)*delay)
            self.assertEqual(tuple(x for x in outputs if x!=ZERO_ACTION),COMMANDS)
            self.assertTrue(all(len(x)==4 and all(-1<=v<=1 for v in x) for x in outputs))

    def test_noise_hold_fixture_covers_full_episode_and_rejects_hold_extension(self):
        self.assertEqual(len(NOISE_OFFSETS_M),51)
        self.assertEqual(len(DROPOUT_DRAWS),51)
        self.assertEqual(len(SENSOR_COMMANDS),50)
        self.assertEqual(DROPOUT_HOLD,3)
        self.assertLess(DROPOUT_DRAWS[3],DROPOUT_PROBABILITY)
        self.assertEqual(HOLD_BLOCKS[0],(2,4,1))  # Trigger3 cannot extend the hold to5.
        for first,last,anchor in HOLD_BLOCKS:
            self.assertEqual(last-first+1,3)
            self.assertLess(anchor,first)
            self.assertNotEqual(NOISE_OFFSETS_M[first],NOISE_OFFSETS_M[anchor])
            self.assertNotEqual(NOISE_OFFSETS_M[last+1],NOISE_OFFSETS_M[anchor])

if __name__=='__main__': unittest.main()
