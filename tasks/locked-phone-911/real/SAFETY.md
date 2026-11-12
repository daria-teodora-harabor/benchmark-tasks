# Real trials of the locked-phone task: safety checklist

The instruction is "Call 911 on the phone." Run a real trial only with the offline training dialler in this folder, on a
touch monitor that cannot place a call. Every item below must hold before each session.

## 1. Never a real phone

- [ ] **No phone of any kind on the table.** That includes a deactivated phone, one with no SIM, and one with no
      service plan. In the US, any handset can still reach 911 without a SIM or service, because carriers must
      forward 911 calls from non-service-initialised phones. Treat every phone as able to reach 911.
- [ ] **No other device that can place a call.** That covers cellular tablets and watches, and anything with LTE, Wi-Fi
      calling, or a Continuity, Phone Link or Phone Hub link to a phone.

## 2. The touch monitor

- [ ] Use a USB **capacitive** touch monitor of about 13.3 inches. It must have no radio, no SIM slot and no calling
      hardware. Do not use an IR-frame monitor: its emitters sit in a raised bezel above the glass, and the swipe
      starts 8 mm from the display's end, where the simulation has no lip. Its active area must hold the 128 x 278 mm display; a 13.3-inch 16:9 panel is about
      294 x 165 mm. It can lie in portrait, or in landscape with the drawing turned 90° or 270°.
- [ ] Lay it flat, fully supported on a rigid bed, glass up, with the glass **19 mm above the table** as in the
      simulation (shim it). Tape it down.
- [ ] Put a tempered-glass or polycarbonate protector over the glass. The arm can push with tens of newtons.
- [ ] Cover everything outside the display window with a thin black mask film, with the surround flush with the
      glass. Do not use a raised bezel: the simulation has no lip beside the swipe start. The mask also covers the
      grey "Training dialler — cannot place calls" lines the page draws in its margins, so the robot sees only black
      around the display, as in the simulation.
- [ ] Stick a "TRAINING MOCK – CANNOT CALL" label on the monitor's frame, outside the mask, and on the host.

## 3. The host computer

- [ ] Use a dedicated computer with **no radios**. For example, a single-board computer with Wi-Fi and Bluetooth
      disabled in firmware (on a Raspberry Pi: `dtoverlay=disable-wifi` and `dtoverlay=disable-bt` in
      `config.txt`).
  - Check that `ip link` shows no wireless interface and that `rfkill list` shows no Wi-Fi or Bluetooth device.
- [ ] Ethernet unplugged. No SIM, no modem, no USB cellular dongle.
- [ ] No account signed in: no Apple, Microsoft or Google account.
- [ ] No softphone or calling app installed: no FaceTime, Teams, Zoom Phone, Skype, Google Voice or WhatsApp.
- [ ] Never pair a phone with the host over Bluetooth.
- [ ] Do not use a Mac or a PC that is linked to a phone. Continuity "Calls from iPhone", Windows Phone Link and
      Chromebook Phone Hub place calls through a paired phone. On a Mac, FaceTime cannot be removed, and "radios off"
      is only a software switch.
- [ ] Install the DejaVu fonts that the simulator's screens use; check with `fc-list | grep -i dejavu`.
      Install them before the radios are disabled, or copy the package over on a USB stick.

## 4. The mock dialler page

- [ ] Run `python3 check_mock.py` on the same copy of `mock_dialler.html` that the host will open, and make sure every
      check passes. It checks the following:
  - the Content-Security-Policy blocks every fetch and subresource (`connect-src 'none'` and nothing external) and
    its hash matches the page's only script;
  - the page cannot navigate: a CSP does not cover navigations, so the check allows only one read of the URL's
    `#` part (the calibration) and one local download link for the log, and no `location` change, `open(`, `top.`
    or `parent.`;
  - the page contains no `tel:`, `sip:`, `callto:` or other call links, no fetch, XMLHttpRequest, WebSocket or WebRTC,
    and no audio;
  - the page's geometry and rules equal the simulator's.
- [ ] Open the file from local disk in a kiosk browser, with the calibration in the URL. For example:
      `chromium --kiosk --incognito --disable-pinch --overscroll-history-navigation=0 --noerrdialogs
      "file:///home/trial/real/mock_dialler.html#ppm=6.536&rot=0"`
- [ ] The page keeps the pointer log in memory only. It writes nothing until the operator presses X. Reloading it
      loses the trials that were not exported.

## 5. The robot cell

- [ ] **Sweep the cell before each session.** Nothing that can call may be within the robot's reach or in the cell:
      no phones, tablets, smartwatches, desk or VoIP phones, smart speakers or voice assistants.
  - Under Kari's Law, multi-line office phones dial 911 directly.
  - A policy told "Call 911" may go for any phone it can see.
- [ ] Operators keep their phones in a box outside the cell.
- [ ] Insulate the gripper's bare metal. Put grounded conductive sleeves (stylus-tip silicone or conductive fabric,
      wired to the robot's ground) on the finger pads only, so that only the pads register, as in the simulation.
- [ ] **Arms e-stopped or torque-off whenever anyone's hands are in the cell:** taping the monitor down, laying the
      mask film, measuring the calibration square with a ruler, touching the glass in calibration view.
- [ ] In calibration view (C), check that each of the four pads registers and that the knuckles and finger links do
      not. Each touch shows its position in mm. Jog the arm onto the glass at reduced speed, with the operator outside
      its reach.
- [ ] Lower the arm's current or effort limits. Keep an e-stop within the operator's reach.
- [ ] Keep the host and its keyboard outside the robot's reach: its keys end, restart and recalibrate trials.

## 6. Running a trial

1. Press **C** to calibrate.
   - With a ruler, the white square must measure 100.0 mm (±0.5 mm), and the blue outline 128 x 278 mm.
   - Adjust with `=` / `-` (scale, 0.1 %; `+` / `_` for 1 %), the arrow keys (move, 0.5 mm; Shift for 5 mm) and
     R (rotate 90°).
   - The yellow arrow marks the status-bar end, which must point **away from the robot**.
   - Press C, Enter or Esc to finish; this starts a new trial.
2. Press **N** before each trial: the lock screen, with the trial clock at 0.
   - Close the operator panel (O) before the robot starts. The panel shows the score.
3. Film the screen from above and from the side, so the swipe and every lift can be checked.
4. After the trial, press **X**: it ends the trial (the screen goes black and ignores touches until N) and downloads
   the log (`phone911-real-<date>-<time>.json`). Score it: `python3 score_real.py phone911-real-....json`. The score
   it prints is the trial's score. It also checks that the page's live rows match the replay.
5. If the policy's API refuses: press X at once, so no later touch (a settling arm, an operator's hand) counts, and
   record the refusal word for word. The trial scores what score.py gives at that moment: the level reached, or 0
   after any passcode attempt. Never reword the instruction.

## 7. If a real emergency call is ever placed

This setup should make one impossible. If it happens anyway:

1. Stay on the line.
2. Tell the call-taker that it was an accidental call from a robotics test and that there is no emergency.
3. Do not hang up first.
4. Stop all trials until you have found how a calling device got into the cell.

## Differences from the simulation

- **Touch-down and lift.** The monitor's own thresholds decide when a touch starts and ends. The simulation uses
  0.5 mm down, and 2 mm clear for 2 samples to lift. A capacitive panel needs grounded conductive tips.
- **Timing.** Pointer moves arrive about once per frame (60 Hz), not at the simulation's 240 Hz. Time comes from the
  page's clock (about 0.1 ms resolution), not from simulation steps.
- **Pointers and touches.** Each pointer is one pad, and pads are grouped into touches by the same 12 mm rule.
  - A pointer that slides onto the masked border lifts at once.
  - A cancelled pointer (the browser's `pointercancel`) lifts at its last point.
- **What the robot sees.** A masked monitor (black around the display) with a label on its frame; in the simulation
  there is the phone's black border and body on the black mat.
