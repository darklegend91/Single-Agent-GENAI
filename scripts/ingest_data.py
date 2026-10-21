"""Explicit production ingestion; never substitutes fake vectors or embeddings."""
import argparse
import json
from app.config import Settings
from app.errors import AppError
from app.rag.store import ingest


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--limit',type=int,default=None,help='Explicit positive sample limit; omitted means full source')
    args=parser.parse_args(argv)
    try:
        result=ingest(Settings(),limit=args.limit)
    except AppError as error:
        print(json.dumps({'error':error.code,'message':error.message})); return 1
    print(json.dumps({'dataset_version':result.snapshot.version,'row_count':result.row_count,'sample_limit':result.sample_limit,'reused':result.reused}))
    return 0


if __name__=='__main__': raise SystemExit(main())
