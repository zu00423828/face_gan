import os
import torch
class BaseModel():
    def name(self):
        return "BaseModel"
    def set_input(self,input):
        self.input=input
    def forward(self):
        pass
    def test(self):
        with torch.no_grad():
            self.forward()
    def get_image_paths(self):
        pass
    def optimize_parmeters(self):
        pass
    def get_current_visuals(self):
        return self.input
    def save(self,label):
        pass
    def save_network(self,network,network_label,epoch_label):
        save_filename='%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(network.cpu().state_dict(), save_path)
        