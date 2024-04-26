import argparse
from rl_zoo3.tracking.tracking_backend import TrackingBackend


def argparse_add_track_arguments(parser: argparse.ArgumentParser) -> None:
    avaiable_backends = TrackingBackend.list_backends()
    if len(avaiable_backends) == 0:
        return

    example = avaiable_backends.pop()
    avaiable_backends.add(example)

    parser.add_argument(
        "--track-backend",
        default=example,
        choices=avaiable_backends,
        help="select ML platform for tracking experiments",
    )

    for bd_str in avaiable_backends:
        backend = TrackingBackend.get_tracker(bd_str)
        backend.argparse_add_arguments(parser)


def argparse_filter_track_arguments(parsed_args) -> None:
    avaiable_backends = TrackingBackend.list_backends()
    if len(avaiable_backends) == 0:
        return

    for bd_str in avaiable_backends:
        if bd_str == parsed_args.track_backend:
            continue

        backend = TrackingBackend.get_tracker(bd_str)
        backend.argparse_del_arguments(parsed_args)
