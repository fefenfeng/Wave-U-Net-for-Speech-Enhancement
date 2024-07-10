import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import torch

from trainer.base_trainer import BaseTrainer
from util.utils import compute_STOI, compute_PESQ
plt.switch_backend('agg')


class Trainer(BaseTrainer):
    """
    重载父类basetrainer，主要重写_train_epoch和validation_epoch方法
    注意这里的两个方法是在循环内的，即每个方法都是在单个epoch进行的方法
    """
    def __init__(
            self,
            config,
            resume: bool,
            model,
            loss_function,
            optimizer,
            train_dataloader,
            validation_dataloader,
    ):
        super(Trainer, self).__init__(config, resume, model, loss_function, optimizer)
        self.train_data_loader = train_dataloader
        self.validation_data_loader = validation_dataloader

    def _train_epoch(self, epoch):
        loss_total = 0.0

        for i, (mixture, clean, name) in enumerate(self.train_data_loader):
            # 数据移到device
            mixture = mixture.to(self.device)
            clean = clean.to(self.device)
            # 清零梯度
            self.optimizer.zero_grad()
            # 前向传播 enhanced就是output
            enhanced = self.model(mixture)
            # 现在是mse_loss
            loss = self.loss_function(clean, enhanced)

            loss.backward()  # 反向传播
            self.optimizer.step()  # 梯度更新参数

            loss_total += loss.item()  # 累计loss

        dl_len = len(self.train_data_loader)  # 获取总共这个epoch中loader中有多少batch
        # 当前平均损失，loss_total / dl_len(batch_num)
        self.writer.add_scalar(f"Train/Loss", loss_total / dl_len, epoch)

    @torch.no_grad()
    def _validation_epoch(self, epoch):
        # 验证过程，验证集评估的表现，并记录响应指标和可视化结果
        # 加载trainer.validation.custom参数
        visualize_audio_limit = self.validation_custom_config["visualize_audio_limit"]
        visualize_waveform_limit = self.validation_custom_config["visualize_waveform_limit"]
        visualize_spectrogram_limit = self.validation_custom_config["visualize_spectrogram_limit"]

        sample_length = self.validation_custom_config["sample_length"]

        # 初始化评估指标列表
        stoi_c_n = []  # clean and noisy
        stoi_c_e = []  # clean and enhanced
        pesq_c_n = []
        pesq_c_e = []

        # loss total
        loss_total = 0.0

        for i, (mixture, clean, name) in enumerate(self.validation_data_loader):
            assert len(name) == 1, "Only support batch size is 1 in enhancement stage."
            name = name[0]
            padded_length = 0  # 用于记录如果音频长度不是sample_length整数倍时，需要填充的长度

            mixture = mixture.to(self.device)  # [1, 1, T]
            clean = clean.to(self.device)  # clean也移到device上

            # 模型输入是固定长度，需要将mixture文件（验证文件本来长度不限）分成许多块
            if mixture.size(-1) % sample_length != 0:
                # 不是整数倍，需要填充
                padded_length = sample_length - (mixture.size(-1) % sample_length)
                mixture = torch.cat([mixture, torch.zeros(1, 1, padded_length, device=self.device)], dim=-1)
            # 切分音频块
            assert mixture.size(-1) % sample_length == 0 and mixture.dim() == 3
            mixture_chunks = list(torch.split(mixture, sample_length, dim=-1))

            enhanced_chunks = []
            for chunk in mixture_chunks:
                # detach()从计算图中分离tensor
                # enhanced_chunks.append(self.model(chunk).detach().cpu())
                enhanced_chunks.append(self.model(chunk).detach())
            # 拼接tensor得到full tensor
            enhanced = torch.cat(enhanced_chunks, dim=-1)  # [1, 1, T]
            # 移除padded填充部分
            enhanced = enhanced if padded_length == 0 else enhanced[:, :, :-padded_length]
            mixture = mixture if padded_length == 0 else mixture[:, :, :-padded_length]

            # 计算loss
            loss = self.loss_function(clean, enhanced)
            loss_total += loss.item()

            # 转成一维numpy数组
            # enhanced = enhanced.reshape(-1).numpy()
            # clean = clean.numpy().reshape(-1)
            enhanced = enhanced.cpu().reshape(-1).numpy()
            clean = clean.cpu().numpy().reshape(-1)
            mixture = mixture.cpu().numpy().reshape(-1)

            assert len(mixture) == len(enhanced) == len(clean)

            # Visualize audio
            if i <= visualize_audio_limit:
                self.writer.add_audio(f"Speech/{name}_Noisy", mixture, epoch, sample_rate=16000)
                self.writer.add_audio(f"Speech/{name}_Enhanced", enhanced, epoch, sample_rate=16000)
                self.writer.add_audio(f"Speech/{name}_Clean", clean, epoch, sample_rate=16000)

            # Visualize waveform
            if i <= visualize_waveform_limit:
                fig, ax = plt.subplots(3, 1)
                for j, y in enumerate([mixture, enhanced, clean]):
                    ax[j].set_title("mean: {:.3f}, std: {:.3f}, max: {:.3f}, min: {:.3f}".format(
                        np.mean(y),
                        np.std(y),
                        np.max(y),
                        np.min(y)
                    ))
                    # librosa.display.waveplot(y, sr=16000, ax=ax[j])  # 旧版本librosa
                    librosa.display.waveshow(y, sr=16000, ax=ax[j])
                plt.tight_layout()
                self.writer.add_figure(f"Waveform/{name}", fig, epoch)

            # Visualize spectrogram, hop length一半overlap一半帧移一半
            noisy_mag, _ = librosa.magphase(librosa.stft(mixture, n_fft=320, hop_length=160, win_length=320))
            enhanced_mag, _ = librosa.magphase(librosa.stft(enhanced, n_fft=320, hop_length=160, win_length=320))
            clean_mag, _ = librosa.magphase(librosa.stft(clean, n_fft=320, hop_length=160, win_length=320))

            if i <= visualize_spectrogram_limit:
                fig, axes = plt.subplots(3, 1, figsize=(6, 6))
                for k, mag in enumerate([noisy_mag, enhanced_mag, clean_mag]):
                    axes[k].set_title(f"mean: {np.mean(mag):.3f}, "
                                      f"std: {np.std(mag):.3f}, "
                                      f"max: {np.max(mag):.3f}, "
                                      f"min: {np.min(mag):.3f}")
                    librosa.display.specshow(librosa.amplitude_to_db(mag), cmap="magma", y_axis="linear", ax=axes[k], sr=16000)
                plt.tight_layout()
                self.writer.add_figure(f"Spectrogram/{name}", fig, epoch)

            # Metric
            stoi_c_n.append(compute_STOI(clean, mixture, sr=16000))
            stoi_c_e.append(compute_STOI(clean, enhanced, sr=16000))
            pesq_c_n.append(compute_PESQ(clean, mixture, sr=16000))
            pesq_c_e.append(compute_PESQ(clean, enhanced, sr=16000))

        get_metrics_ave = lambda metrics: np.sum(metrics) / len(metrics)
        self.writer.add_scalars(f"Metric/STOI", {
            "Clean and noisy": get_metrics_ave(stoi_c_n),
            "Clean and enhanced": get_metrics_ave(stoi_c_e)
        }, epoch)
        self.writer.add_scalars(f"Metric/PESQ", {
            "Clean and noisy": get_metrics_ave(pesq_c_n),
            "Clean and enhanced": get_metrics_ave(pesq_c_e)
        }, epoch)

        # 记录平均损失到TensorBoard
        dl_len = len(self.validation_data_loader)
        self.writer.add_scalar(f"Validation/Loss", loss_total / dl_len, epoch)

        score = (get_metrics_ave(stoi_c_e) + self._transform_pesq_range(get_metrics_ave(pesq_c_e))) / 2
        return score
