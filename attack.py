import torch
import torch.nn.functional as F
import numpy as np

class FGSM:
    def __init__(self, model, epsilon=0.3):
        self.model = model
        self.epsilon = epsilon
        self.model.eval()

    def attack(self, x, y):
        x = x.clone().detach().requires_grad_(True)
        outputs = self.model(x)
        loss = F.cross_entropy(outputs, y)
        self.model.zero_grad()
        loss.backward()
        
        grad = x.grad.data
        x_adv = x + self.epsilon * grad.sign()
        x_adv = torch.clamp(x_adv, 0, 1)
        return x_adv.detach()

class BlackBoxAttack:
    def __init__(self, model, device, max_queries=5000, epsilon=0.3, lr=0.05, sigma=0.01, n_samples=100, targeted=False):
        """
        Initializes the NES black-box attack.
        :param targeted: Whether the attack is targeted.
        """
        self.model = model
        self.device = device
        self.max_queries = max_queries
        self.epsilon = epsilon
        self.lr = lr
        self.sigma = sigma
        self.n_samples = n_samples
        self.targeted = targeted
        self.model.eval()

    def estimate_gradient(self, x, target_label):
        """
        Estimates the gradient using NES.
        """
        grad = torch.zeros_like(x)
        queries = 0
        
        with torch.no_grad():
            for _ in range(self.n_samples // 2):
                noise = torch.randn_like(x).to(self.device)
                
                x_plus = x + self.sigma * noise
                x_minus = x - self.sigma * noise
                
                out_plus = F.softmax(self.model(x_plus), dim=1)
                out_minus = F.softmax(self.model(x_minus), dim=1)
                queries += 2
                
                prob_plus = out_plus[0, target_label]
                prob_minus = out_minus[0, target_label]
                
                grad += (prob_plus - prob_minus) * noise
                
        return grad / (self.n_samples * self.sigma), queries

    def attack(self, x_orig, y_true_or_target):
        """
        Performs the black-box attack.
        :param y_true_or_target: True label (untargeted) or Target label (targeted).
        """
        x_orig = x_orig.to(self.device)
        x_adv = x_orig.clone().detach()
        
        total_queries = 0
        success = False
        
        # Initial check
        with torch.no_grad():
            initial_out = self.model(x_adv)
            initial_pred = torch.argmax(initial_out, dim=1).item()
            total_queries += 1
            
            if self.targeted:
                if initial_pred == y_true_or_target:
                    return x_adv, 1, True, initial_pred
            else:
                if initial_pred != y_true_or_target:
                    return x_adv, 1, True, initial_pred

        while total_queries < self.max_queries:
            # Estimate gradient
            # For untargeted: we minimize prob of true label
            # For targeted: we maximize prob of target label
            target_label = y_true_or_target
            grad, queries = self.estimate_gradient(x_adv, target_label)
            total_queries += queries
            
            # Gradient update direction
            if self.targeted:
                # Maximize target prob -> move in direction of gradient
                x_adv = x_adv + self.lr * torch.sign(grad)
            else:
                # Minimize true label prob -> move away from gradient
                x_adv = x_adv - self.lr * torch.sign(grad)
            
            # Constraints
            x_adv = torch.clamp(x_adv, x_orig - self.epsilon, x_orig + self.epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
            
            # Success check
            with torch.no_grad():
                out = self.model(x_adv)
                pred = torch.argmax(out, dim=1).item()
                total_queries += 1
                
                if self.targeted:
                    if pred == y_true_or_target:
                        return x_adv, total_queries, True, pred
                else:
                    if pred != y_true_or_target:
                        return x_adv, total_queries, True, pred
                    
        return x_adv, total_queries, False, pred

if __name__ == "__main__":
    # Test stub
    pass
