import torch
import torch.nn as nn
import torch.optim as optim

BEST_EPOCH_DICT = 'best_epoch_dict'
BEST_OPTIMIZER_DICT = 'best_optimizer_dict'
BEST_MODEL_DICT = 'best_model_dict'
BEST_PERFORMANCE_DICT = 'best_performance_dict'


class ModelRecorder:

    def __init__(self, save_file=None, optimizer: optim.Optimizer = None, comparator_dict=None, save_record=False,
                 summary_writer=None):
        self.save_file = save_file
        self.optimizer = optimizer
        self.save_record = save_record
        self.summary_writer = summary_writer
        if comparator_dict is None:
            comparator_dict = {}
        self.best_epoch_dict = {}
        self.best_model_dict = {}
        self.comparator_dict = comparator_dict
        self.best_performance_dict = {}
        self.best_optimizer_dict = {}
        self.performance_record = []

    @staticmethod
    def resume_model(model: nn.Module, ckpt_file: str, from_measurement: str, print_detail=True):
        state_dict = ModelRecorder.load_model_state_dict(ckpt_file, from_measurement, print_detail)
        model.load_state_dict(state_dict)

    @staticmethod
    def load_model_state_dict(ckpt_file: str, from_measurement: str, print_detail=True):
        ckpt = torch.load(ckpt_file, 'cpu')
        if print_detail:
            print("load from epoch: {},".format(ckpt[BEST_EPOCH_DICT][from_measurement]) + \
                  "{}".format(from_measurement, ckpt[BEST_PERFORMANCE_DICT][from_measurement]))
        return ckpt[BEST_MODEL_DICT][from_measurement]

    def resume(self, model, optimizer, from_measurement: str, ckpt_file=None, print_detail=True):
        if ckpt_file is not None:
            ckpt = torch.load(ckpt_file)
        else:
            ckpt = torch.load(self.save_file)
        self.best_performance_dict = ckpt[BEST_PERFORMANCE_DICT]
        self.best_model_dict = ckpt[BEST_MODEL_DICT]
        self.best_optimizer_dict = ckpt[BEST_OPTIMIZER_DICT]
        self.best_epoch_dict = ckpt[BEST_EPOCH_DICT]
        if print_detail:
            print("load from epoch: {},".format(ckpt[BEST_EPOCH_DICT][from_measurement]) + \
                  "{}: {}".format(from_measurement, ckpt[BEST_PERFORMANCE_DICT][from_measurement]))
        model.load_state_dict(self.best_model_dict[from_measurement])
        optimizer.load_state_dict(self.best_optimizer_dict[from_measurement])
        return self.best_epoch_dict[from_measurement]

    @DeprecationWarning
    def load(self, ckpt_file: str, from_measurement: str):
        """
        Load model `state_dict` from ckpt_file
        :param ckpt_file: path to ckpt_file
        :param from_measurement: get the best model dict using the specified measurement
        :return: (Best model's state_dict, Epoch, Optimizer state_dict, Performance by 'from_measurement')
        """
        ckpt = torch.load(ckpt_file)
        self.best_performance_dict = ckpt[BEST_PERFORMANCE_DICT]
        self.best_model_dict = ckpt[BEST_MODEL_DICT]
        self.best_optimizer_dict = ckpt[BEST_OPTIMIZER_DICT]
        self.best_epoch_dict = ckpt[BEST_EPOCH_DICT]

        return self.best_model_dict[from_measurement], self.best_epoch_dict[from_measurement], self.best_optimizer_dict[
            from_measurement], self.best_performance_dict[from_measurement],

    def add(self, epoch, model: torch.nn.Module, performance_dict: dict, addition_info=None):
        self.performance_record.append(performance_dict)
        # record performance
        for key in performance_dict:
            # write to tensorboard
            if self.summary_writer is not None:
                self.summary_writer.add_scalar(str(key), performance_dict[key], global_step=epoch)
            # record the best
            curr_best = False
            if key not in self.best_performance_dict:
                curr_best = True
            else:
                is_better = self.__default_comparator
                if key in self.comparator_dict:
                    is_better = self.comparator_dict[key]
                if is_better(performance_dict[key], self.best_performance_dict[key]):
                    curr_best = True
            if curr_best:
                # save as cpu tensor for faster preview (e.g. preview in Ipython)
                model_state_dict = self.__get_cpu_model_state_dict(model)
                self.best_performance_dict[key] = performance_dict[key]
                self.best_epoch_dict[key] = epoch
                self.best_model_dict[key] = model_state_dict
                self.best_optimizer_dict[key] = self.optimizer.state_dict()
        torch.save(dict(epoch=epoch,
                        best_epoch_dict=self.best_epoch_dict,
                        best_model_dict=self.best_model_dict,
                        best_performance_dict=self.best_performance_dict,
                        addition_info=addition_info,
                        best_optimizer_dict=self.best_optimizer_dict), self.save_file)

        if self.save_record:
            # TODO
            import sys
            print("record save is not implemented yet!", file=sys.stderr)
            pass

    @staticmethod
    def __get_cpu_model_state_dict(model):
        if isinstance(model, torch.nn.DataParallel):
            model = model.module
        assert isinstance(model, torch.nn.Module)
        model_state_dict = model.state_dict()
        for k in model_state_dict:
            model_state_dict[k] = model_state_dict[k].cpu()
        return model_state_dict

    @staticmethod
    def __default_comparator(x, y):
        # x is better than y
        return x > y

    @staticmethod
    def __save_record(epoch, c_prec, net: nn.Module):
        state_dict = net.state_dict()
        # torch.save(state_dict, f"{config.ckpt_record_folder}/epoch{epoch}_{c_prec:.2f}.pth")
        raise NotImplementedError()

    def print_best_stat(self):
        for key in self.best_performance_dict:
            print('best {}: {} (at epoch {})'.format(key, self.best_performance_dict[key], self.best_epoch_dict[key]))

    def print_curr_stat(self, print_best=True):
        curr_performance_dict = self.performance_record[-1]
        for key in curr_performance_dict:
            print('curr {}: {} '.format(key, curr_performance_dict[key]))
        if print_best:
            self.print_best_stat()
