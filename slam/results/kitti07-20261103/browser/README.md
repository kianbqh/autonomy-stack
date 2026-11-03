# Recorded SLAM browser checks

I use these captures to check that the viewer receives the actual exported trajectory and keeps each failed attempt distinct. All checks use Chromium156.0.8078.4 with headless ANGLE SwiftShader. These are functional and visual checks; no FPS or performance result is collected.

The first pass at viewer61d checks the complete1101-frame KITTI replay, the originalCA9 real27-frame failed prefix and the originalCA9 synthetic2-frame degeneracy prefix. Served JSON hashes, all original IDs/times/native poses, normalizedGT and actual3D trajectory/loop buffers match. The loop layers contain2 accepted,1852 rejected and4581 temporally excluded decisions. All three cases pass without page errors; original failure messages and unavailable optimization remain visible.

That pass exposed a default-camera clipping issue in the full trajectory. The focused pass at viewera513 retains the same native61d replay and fits the full3D bounds to the viewport. Actual GL projection/model-view matrices put all3303 track points inside the clip volume in desktop1440×1000 and narrow480×1000 views. The complete route is visible in both focused screenshots. The two historical failure cases were not rerun after this camera-only change.

The first focused harness attempt accidentally matched the mainHTML query URL instead of the JSON request. It failed before geometry/screenshot checks. Its original receipt/logs/source are retained; the corrected exact-URL predicate passes in a fresh output directory with the same build/server/data. Both dedicated servers and browser contexts were closed afterward.

Final screenshots: [desktop](focused-fit/desktop-end.png), [narrow](focused-fit/narrow-end.png). Raw [functional receipt](functional/browser-acceptance.json), [focused receipt](focused-fit/fit-regression.json) and file hashes inMANIFEST.json retain separate viewer/native identities. Earlier full07 screenshots underfunctional/ show the pre-fit framing; the focused screenshots supersede that visual result. Recorded harness copies retain the original local paths and fixed artifact identities.

KITTI odometry by Geiger et al.; CC BY-NC-SA3.0. Images here show derived trajectory replay rather than raw sensor clouds. Synthetic failure evidence remains explicitly synthetic.
