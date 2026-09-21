# assets

`impulseresponse.wav` is the room impulse response used by the reverb effect (`src/effects.py` keeps its 1.01 s to
1.30 s portion, normalised to unit energy). It is **not included** in this repository because its source and license
have not been documented yet.

Put the file here (`assets/impulseresponse.wav`) or point `SSS_IMPULSE_RESPONSE` at it. If you replace it with a different
room impulse response, the reverb strength and therefore every reverb result will change.

TODO before publishing: record where this impulse response came from, its license, and its approximate decay time,
then either add the file here or link to it.
