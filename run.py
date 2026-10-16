"""Project entry point. Run --help for the safe review and final-evaluation paths."""
import argparse
import json
from voice_review.experiment import prepare, train, evaluate_final, predict, freeze

def main():
    parser = argparse.ArgumentParser(description='Academic Voice Disorder Recognizer')
    parser.add_argument('command', choices=['prepare','train','report','predict','freeze','evaluate-final','robustness'])
    parser.add_argument('--audio', help='Path to a WAV/FLAC file, or a WFDB .hea file')
    parser.add_argument('--validation-demo', action='store_true', help='Use the verified speaker-held-out fold models for an unchanged development recording')
    parser.add_argument('--confirm-final', action='store_true', help='Open the untouched test set exactly once')
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare()
    elif args.command == 'train':
        train()
    elif args.command == 'report':
        from voice_review.reporting import build_report
        build_report()
    elif args.command == 'freeze':
        freeze()
    elif args.command == 'robustness':
        from voice_review.robustness import run_robustness
        run_robustness()
    elif args.command == 'predict':
        if not args.audio:
            parser.error('--audio is required for prediction')
        print(json.dumps(predict(args.audio, validation=args.validation_demo), indent=2))
    else:
        evaluate_final(confirm=args.confirm_final)
        from voice_review.reporting import build_report
        build_report()

if __name__ == '__main__':
    main()
