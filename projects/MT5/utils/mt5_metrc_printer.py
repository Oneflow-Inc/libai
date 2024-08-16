import datetime
import logging
import time

from libai.utils.events import EventWriter, get_event_storage


class MT5MetricPrinter(EventWriter):
    """
    Print **MT5** metrics to the terminal, including
    iteration time, ETA, memory, all losses, and the learning rate.
    It also applies smoothing using a window of 20 elements.
    It's meant to print MT5 metrics in MT5 ways.
    To print something in more customized ways, please implement a similar printer by yourself.
    """

    def __init__(self, batch_size, max_iter, log_period):
        """
        Args:
            max_iter (int): the maximum number of iterations to train.
                Used to compute ETA.
        """
        self.logger = logging.getLogger("libai." + __name__)
        self._batch_size = batch_size
        self._max_iter = max_iter
        self._last_write = None
        self._log_period = log_period

    def write(self):
        storage = get_event_storage()
        iteration = storage.iter
        consumed_samples = storage.samples

        try:
            done_tokens = storage.history("done_tokens").avg(self._log_period)
            token_time = storage.history("time").avg(self._log_period)
        except KeyError:
            done_tokens = None

        try:
            correct_tokens = storage.history("correct_tokens").avg(self._log_period)
            denominator = storage.history("denominator").avg(self._log_period)
            acc_mlm = correct_tokens / denominator
        except KeyError:
            acc_mlm = None

        if iteration == self._max_iter:
            # This hook only reports training progress (loss, ETA, etc) but not other data,
            # therefore do not write anything after training succeeds, even if this method
            # is called.
            return

        try:
            data_time = storage.history("data_time").avg(self._log_period)
        except KeyError:
            # they may not exist in the first few iterations (due to warmup)
            # or when SimpleTrainer is not used
            data_time = None

        eta_string = None
        try:
            iter_time = storage.history("time").global_avg()
            eta_seconds = storage.history("time").median(1000) * (self._max_iter - iteration - 1)
            storage.put_scalar("eta_seconds", eta_seconds, smoothing_hint=False)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
        except KeyError:
            iter_time = None
            # estimate eta on our own - more noisy
            if self._last_write is not None:
                estimate_iter_time = (time.perf_counter() - self._last_write[1]) / (
                    iteration - self._last_write[0]
                )
                eta_seconds = estimate_iter_time * (self._max_iter - iteration - 1)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
            self._last_write = (iteration, time.perf_counter())

        try:
            lr = "{:.2e}".format(storage.history("lr").latest())
        except KeyError:
            lr = "N/A"

        max_mem_mb = None

        # NOTE: max_mem is parsed by grep in "dev/parse_results.sh"
        self.logger.info(
            " {eta}  {iter}  {sample}  {losses}  {time}  {data_time}  {tpt}  lr: {lr}  {memory} "
            " {tokens_speed}  {acc_mlm}".format(
                eta=f"eta: {eta_string}" if eta_string else "",
                iter=f"iteration: {iteration}/{self._max_iter}",
                sample=f"consumed_samples: {consumed_samples}",
                losses="  ".join(
                    [
                        "{}: {:.4g}".format(k, v.median(200))
                        for k, v in storage.histories().items()
                        if "loss" in k
                    ]
                ),
                time="time: {:.4f} s/iter  ".format(iter_time) if iter_time is not None else "",
                data_time="data_time: {:.4f} s/iter".format(data_time)
                if data_time is not None
                else "",
                tpt="total_throughput: {:.2f} samples/s".format(self._batch_size / iter_time)
                if iter_time is not None
                else "",
                lr=lr,
                memory="max_mem: {:.0f}M".format(max_mem_mb) if max_mem_mb is not None else "",
                tokens_speed="tokens_throughput: {:.4f} tokens/s".format(done_tokens / token_time)
                if done_tokens is not None
                else "",
                acc_mlm="acc_mlm: {:.4f}".format(acc_mlm) if acc_mlm is not None else "",
            )
        )
