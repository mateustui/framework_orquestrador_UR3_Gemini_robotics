from __future__ import annotations

import threading
import time
from queue import Queue

from orquestrador.adapters.gui.window import UnifiedGUI
from orquestrador.app.simulation import simulation_thread


def main() -> None:
    cmd_queue: Queue = Queue()
    log_queue: Queue = Queue()
    frame_queue: Queue = Queue(maxsize=2)
    running = [True]

    sim_thread = threading.Thread(
        target=simulation_thread,
        args=(cmd_queue, log_queue, frame_queue, running),
        daemon=True,
    )
    sim_thread.start()

    gui = UnifiedGUI(cmd_queue, log_queue, frame_queue, running)
    gui.run()

    running[0] = False
    time.sleep(0.3)


if __name__ == "__main__":
    main()
