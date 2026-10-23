"""Offline prediction with the same service as the Streamlit website."""
import argparse,json
from voice_review.release_web import predict_voice

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('audio_file');args=parser.parse_args()
    try:
        result=predict_voice(args.audio_file)
        print(json.dumps({k:v for k,v in result.items() if k!='audio'},indent=2))
    except (ValueError,OSError) as exc:parser.exit(2,str(exc)+'\n')
