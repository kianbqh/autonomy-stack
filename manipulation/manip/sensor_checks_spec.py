"""Predeclared engineering fixtures, not measurements or another sensor simulator.

The runtime checks send these commands through the actual installed Fetch wrapper.
The two literal expected timelines also include the final drain commands, so every
nonzero command is checked after the configured delay. No production queue helper
is called to construct the expected output.
"""
ZERO_ACTION=(0.,0.,0.,0.)
COMMANDS=(
    (.11,0.,.02,0.), (-.07,.04,0.,0.), (.03,-.09,.01,0.), (0.,.02,-.05,0.),
    (.06,-.03,0.,0.), (-.02,.05,.02,0.), (.08,0.,-.01,0.), (0.,-.04,.03,0.),
)
FIFO_COMMANDS=COMMANDS+(ZERO_ACTION,ZERO_ACTION)
FIFO_EXPECTED={
    1:(ZERO_ACTION,COMMANDS[0],COMMANDS[1],COMMANDS[2],COMMANDS[3],COMMANDS[4],
       COMMANDS[5],COMMANDS[6],COMMANDS[7],ZERO_ACTION),
    2:(ZERO_ACTION,ZERO_ACTION,COMMANDS[0],COMMANDS[1],COMMANDS[2],COMMANDS[3],
       COMMANDS[4],COMMANDS[5],COMMANDS[6],COMMANDS[7]),
}
# Position offsets are already in metres; these are injected engineering samples,
# not draws used to estimate Gaussian sensor parameters or calibration accuracy.
NOISE_PATTERN_M=((.003,-.002,.001),(-.004,.001,.002),(.002,.003,-.001),
                 (.001,-.003,.004),(-.002,.004,-.003))
NOISE_OFFSETS_M=tuple(NOISE_PATTERN_M[i%5] for i in range(51))
DROPOUT_TRIGGER_STEPS=(2,3,7,12,25,47)
DROPOUT_DRAWS=tuple(.25 if i in DROPOUT_TRIGGER_STEPS else .75 for i in range(51))
DROPOUT_PROBABILITY=.5
DROPOUT_HOLD=3
# Inclusive output-step blocks -> visible source step. A trigger during an active
# hold (step3) must not extend that hold. All unlisted steps expose their own pose.
HOLD_BLOCKS=((2,4,1),(7,9,6),(12,14,11),(25,27,24),(47,49,46))
SENSOR_COMMANDS=tuple(COMMANDS[i%8] for i in range(50))
