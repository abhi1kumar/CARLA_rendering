from fire import Fire

import src


if __name__ == '__main__':
    Fire({
        'scrape': src.sim_nuscenes.scrape,
    })
