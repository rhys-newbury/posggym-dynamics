"""Script for manually running and visualizing GridGenerator functionality."""
import argparse
import random
import sys

from posggym.envs.grid_world.core import GridGenerator


def _generate_mask(args):
    rng = random.Random(None) if args.seed is None else random.Random(args.seed + 1)

    mask = set()
    for _ in range(rng.randint(1, 2 * args.width + args.height)):
        x = rng.randint(0, args.width - 1)
        y = rng.randint(0, args.height - 1)
        mask.add((x, y))

    return mask


def main(args):
    """Run."""
    mask = _generate_mask(args) if args.use_random_mask else set()

    if args.max_obstacle_size is None:
        max_obstacle_size = max(1, min(args.width, args.height) // 4)
    else:
        max_obstacle_size = args.max_obstacle_size

    seed = args.seed
    while True:
        n = -1
        while n < 0:
            try:
                n = int(input("Select max_num_obstacles (Ctrl-C to exit): "))
            except ValueError:
                pass
            except KeyboardInterrupt:
                print()
                sys.exit(1)

        grid_gen = GridGenerator(
            args.width,
            args.height,
            mask,
            max_obstacle_size,
            max_num_obstacles=n,
            ensure_grid_connected=False,
            seed=seed,
        )
        seed += 1
        grid = grid_gen.generate()
        print(grid_gen.get_grid_str(grid))

        if args.check_grid_connectedness:
            components = grid.get_connected_components()
            if len(components) == 1:
                print("Grid fully connected")
            else:
                print(f"Grid divided into {len(components)} separate parts.")
                print("Connecting grid")
                grid = grid_gen.connect_grid_components(grid)
                print("Et voila")
                print(grid_gen.get_grid_str(grid))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("width", type=int, help="Width of grid")
    parser.add_argument("height", type=int, help="Height of grid")
    parser.add_argument(
        "--use_random_mask",
        action="store_true",
        help="Use a random mask for generated grids",
    )
    parser.add_argument(
        "--max_obstacle_size",
        type=int,
        default=None,
        help="Max size of obstacle. If None then uses min(width, height) // 4",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random Seed")
    parser.add_argument(
        "--check_grid_connectedness",
        action="store_true",
        help="Also check for grid connectedness",
    )
    main(parser.parse_args())
