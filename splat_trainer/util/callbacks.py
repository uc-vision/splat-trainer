from hydra.experimental.callback import Callback
from omegaconf import DictConfig, OmegaConf
from hydra.core.utils import JobReturn, JobStatus
from typing import Any
from collections import namedtuple, defaultdict
import pandas as pd
import os

class Average_result(Callback):
    def __init__(self, output_dir: str, sweep_params: DictConfig) -> None:
        self.output_dir = output_dir
        self.config_fields = None
        self.Config = None
        self.Result = None
        self.data = []

        if sweep_params:
            self.create_config_namedtuple(sweep_params)


    def create_config_namedtuple(self, params: DictConfig):
        self.config_fields = [self.sanitize_field_name(param) for param in params.keys()]
        self.Config = namedtuple('Config', self.config_fields)


    def on_job_end(self, config: DictConfig, job_return: JobReturn, **kwargs: Any) -> None:
        job_num = config.hydra.job.num
        print(f"Job {job_num} ended. Collecting job result...")

        if OmegaConf.select(config, "hydra.overrides.task"):
            overrides = config.hydra.overrides.task
            override_dict = {self.sanitize_field_name(override.split('=')[0]): override.split('=')[1] for override in overrides}
            if not self.config_fields or self.Config is None:
                self.create_config_namedtuple(overrides)

            config_instance_values = [override_dict.get(field, None) for field in self.config_fields]
            config_instance = self.Config(*config_instance_values)

        if job_return.return_value is not None:
            if self.Result is None: 
                self.Result = namedtuple('Result', job_return.return_value.keys())
            result_instance = self.Result(**job_return.return_value)

        self.data.append((job_num, config_instance._asdict(), result_instance._asdict()))

        with open("data.yaml", "w") as f:
            OmegaConf.save(OmegaConf.create({**config_instance._asdict(), **result_instance._asdict()}), f)


    def on_multirun_end(self, config: DictConfig, **kwargs: Any) -> None:
        "Calculating average results..."
        results = defaultdict(list)
        for job_num, config_instance, result_instance in self.data:
            config_key = tuple(value for name, value in config_instance.items() if name != 'test_scene')
            results[config_key].append(result_instance)

        averaged_results = {}
        for config_key, result_list in results.items():
            result_sums = defaultdict(float)
            result_counts = len(result_list)

            for result_instance in result_list:
                for field_name, value in result_instance.items():
                    result_sums[field_name] += float(value)
        
            averaged_results[config_key] = {
                field_name: result_sums[field_name] / result_counts for field_name in result_sums
            }

        for config_key, averages in averaged_results.items():
            avg_str = ", ".join([f"{key}={value}" for key, value in averages.items()])
            config_str = ", ".join([f"{name}={value}" for name, value in zip(config_instance.keys(), config_key)])
            print(f"Config ({config_str}) -> {avg_str}")

        print("averaged_results: ", averaged_results)
            
        filename = "average_results.csv"
        self.save_to_csv(averaged_results, filename)

    def sanitize_field_name(self, name: str) -> str:
        return name.replace('.', '__')

    def restore_field_name(self, name: str) -> str:
        return name.replace('__', '.')


    def save_to_csv(self, averaged_results, filename):
        data_rows = []

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        output_file = os.path.join(self.output_dir, filename)

        config_fields = [self.restore_field_name(field) for field in self.Config._fields if field != 'test_scene']

        avg_fields = [self.restore_field_name(field) for field in next(iter(averaged_results.values())).keys()]
        
        columns = list(config_fields) + list(avg_fields)

        for config_key, averages in averaged_results.items():
            row = list(config_key) + list(averages.values())
            data_rows.append(row)

        df = pd.DataFrame(data_rows, columns=columns)
        df.to_csv(output_file, index=False)

        print(f"Averaged results saved to {output_file}")





