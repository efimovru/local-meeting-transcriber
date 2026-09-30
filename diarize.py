"""Run the offline pyannote pipeline in a separate process."""

import json
from pathlib import Path
import sys


def speaker_turns(output):
    diarization = getattr(output, "exclusive_speaker_diarization", None)
    if diarization is None:
        diarization = output.speaker_diarization
    return [{"start": turn.start, "end": turn.end, "speaker": speaker}
            for turn, speaker in diarization]


def main(model_dir, audio_path):
    from pyannote.audio import Pipeline

    pipeline = Pipeline.from_pretrained(Path(model_dir))
    output = pipeline(audio_path)
    print(json.dumps(speaker_turns(output)))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
