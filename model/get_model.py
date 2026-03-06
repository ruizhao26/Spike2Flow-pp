from model.spike2flow_pp import spike2flow_pp

def get_model(args, model_name):
    if model_name == 'spike2flow_pp':
        model = spike2flow_pp.Spike2FlowPP(args, input_len=25)
    
    return model