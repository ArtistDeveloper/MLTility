# NOTE: 함수 모듈을 불러오는 식으로 우선 구현. 필요 시 클래스로 확장 예정
# TODO: 모델 학습 매개변수 저장도 필요하면 할 수 있도록 구현

import os
import torch


def save_model_wegights(model, current_epoch, save_path, loss, file_name=None, optimizer=None):
    """
    모델을 재학습시킬 경우는 optimizer도 같이 저장시켜야 합니다.
    """
    
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    
    checkpoint = {
        'epoch': current_epoch,
        'model_state_dict': model.state_dict(),
        'loss': loss
    }
    
    if optimizer:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        file_name = file_name if file_name else f"{current_epoch}_epoch_model_opt.pth"
    else:
        file_name = file_name if file_name else f"{current_epoch}_epoch_model.pth"
    
    file_path = os.path.join(save_path, file_name)
    

    # 이미 파일이 있는 경우, 한 번은 다른 이름으로 다시 저장합니다.
    if not os.path.exists(file_path):
        torch.save(checkpoint, file_path)
    else:
        print(f"{file_path}는 이미 파일이 존재합니다. 다른 이름으로 저장합니다.")
        file_name = f"{current_epoch}_epoch_duplicate_model.pth"
        torch.save(checkpoint, os.path.join(save_path, file_name))


def load_model(model: torch.nn.Module, weights_path, optimizer=None, log=True):
    """
    log = True로 설정하면 해당 함수 내부의 출력문을 사용.
    """
    
    checkpoint = torch.load(weights_path)
    
    # NOTE: GPU를 병렬로 사용한 경우 dictionary의 key이름앞에 'module.'이 붙어 load할 때 key값 매핑이 안되는 경우가 있다. strict=False는 그를 위함.
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if log:
        print(f"checkpoint[epoch]: {checkpoint['epoch']}")
        print(f"checkpoint[loss]: {checkpoint['loss']}")
    
    return model, optimizer, checkpoint
