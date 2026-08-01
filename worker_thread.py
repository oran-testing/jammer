from worker_thread import WorkerThread
import os


class jammer(WorkerThread):
    def start(self):
        self.config.image_name = "ghcr.io/oran-testing/jammer"
        self.cleanup_old_containers()
        self.setup_env()
        self.setup_networks()

        self.config.container_volumes[self.config.config_file] = {
            "bind": "/jammer.yaml", "mode": "ro"
        }

        config_dir = os.path.dirname(os.path.abspath(self.config.config_file))
        self.config.container_volumes[config_dir] = {
            "bind": "/jammer", "mode": "rw"
        }

        self.config.host_network = True

        self.setup_volumes()
        self.start_container()
