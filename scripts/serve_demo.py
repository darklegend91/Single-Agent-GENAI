"""Start the complete local app using only generated synthetic customers."""

import argparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    import uvicorn
    from app.main import create_app
    from scripts.demo_support import create_demo_runtime
    with create_demo_runtime() as runtime:
        print(f"Synthetic demo: http://127.0.0.1:{args.port}", flush=True)
        print("Generated synthetic customers; deterministic demo provider. No external model or vector service required.", flush=True)
        uvicorn.run(create_app(runtime.settings, runtime.services), host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
