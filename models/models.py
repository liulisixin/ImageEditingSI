def create_model(opt):
    if opt.model_name == 'one_decoder':
        from .one_decoder_model import OneDecoderModel
        model = OneDecoderModel(opt)
    elif opt.model_name == 'three_decoder':
        from .three_decoder_model import ThreeDecoderModel
        model = ThreeDecoderModel(opt)
    else:
        raise Exception("Can not find model!")

    print("model [%s] was created" % (model.model_names))
    return model
