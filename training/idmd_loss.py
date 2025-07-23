import torch

from idmd.utils.pfgmpp_utils import sample_from_posterior


class EDMLoss:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5, D="inf"):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        self.D = D

    def primal_loss(self, *, net, images, labels=None):
        noisy_images, sigma, weight = self._prepare_noisy_images_and_sigma_and_weight(images)
        loss = self._primal_loss(net=net, noisy_images=noisy_images, sigma=sigma, labels=labels)
        return loss * weight

    def distillation_loss(self, *, teacher_net, student_net, images, labels=None):
        noisy_images, sigma, weight = self._prepare_noisy_images_and_sigma_and_weight(images)
        teacher_loss = self._primal_loss(net=teacher_net, noisy_images=noisy_images, sigma=sigma, labels=labels)
        student_loss = self._primal_loss(net=student_net, noisy_images=noisy_images, sigma=sigma, labels=labels)
        return (teacher_loss - student_loss) * weight

    def _primal_loss(self, *, net, noisy_images, sigma, labels=None):
        denoised_images = net(noisy_images, sigma, labels)
        loss = ((denoised_images - noisy_images) ** 2)
        return loss

    def _prepare_noisy_images_and_sigma_and_weight(self, images):
        rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        noisy_images = sample_from_posterior(images=images, sigma=sigma, D=self.D)
        return noisy_images, sigma, weight    
