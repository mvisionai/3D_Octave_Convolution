from tensorflow.python.keras import layers
from oct_conv3d import OctConv3D
from tensorflow.python.keras.models import Model

def _create_normal_residual_block(inputs, ch, N):
    # adujust channels
    x = layers.Conv3D(ch, 3, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    # Conv with skip connections
    for i in range(N-1):
        skip = x
        x = layers.Conv3D(ch, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Conv3D(ch, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Add()([x, skip])
    return x

def _create_octconv_residual_block(inputs, ch, N=2, alpha=0.5):
    # adjust channels
    high, low = OctConv3D(filters=ch, alpha=alpha)(inputs)
    high = layers.BatchNormalization()(high)
    high = layers.Activation("relu")(high)
    low = layers.BatchNormalization()(low)
    low = layers.Activation("relu")(low)
    # OctConv with skip connections
    for i in range(N-1):
        skip_high, skip_low = [high, low]

        high, low = OctConv3D(filters=ch, alpha=alpha)([high, low])
        high = layers.BatchNormalization()(high)
        high = layers.Activation("relu")(high)
        low = layers.BatchNormalization()(low)
        low = layers.Activation("relu")(low)

        high, low = OctConv3D(filters=ch, alpha=alpha)([high, low])
        high = layers.BatchNormalization()(high)
        high = layers.Activation("relu")(high)
        low = layers.BatchNormalization()(low)
        low = layers.Activation("relu")(low)

        high = layers.Add()([high, skip_high])
        low = layers.Add()([low, skip_low])
    return [high, low]

def create_normal_wide_resnet(N=3):
    """
    Create vanilla conv Wide ResNet (N=4, k=10)
    """
    # input
    input = layers.Input((212,260,260,1))
    x = layers.Conv3D(filters=16,kernel_size=6,strides=3,padding="same")(input)
    x = layers.AveragePooling3D(pool_size=3,strides=2)(x)
    # 1st block
    x = _create_normal_residual_block(x, 16, N)
    # 2nd block
    x = layers.AveragePooling3D(pool_size=3,strides=2)(x)
    x = _create_normal_residual_block(x, 32, N)
    # 3rd block
    x = layers.AveragePooling3D(pool_size=3,strides=2)(x)
    x = _create_normal_residual_block(x, 64, N)
    # FC
    x = layers.GlobalAveragePooling3D()(x)
    x = layers.Dense(2, activation="softmax")(x)

    model = Model(input, x)
    return model

def create_octconv_wide_resnet(alpha, N=2):
    """
    Create OctConv Wide ResNet(N=4, k=10)
    """
    # Input
    input = layers.Input((212,260,260,1))

    # downsampling for lower

    x=layers.Conv3D(filters=8,kernel_size=5,strides=2)(input)
    low = layers.AveragePooling3D(2)(x)

    # 1st block
    high, low = _create_octconv_residual_block([x, low], 8, N, alpha)
    high,low =  octave_attention_block([high, low],input_channels=16)
    # 2nd block
    high = layers.AveragePooling3D(2)(high)
    low = layers.AveragePooling3D(2)(low)


    high, low = _create_octconv_residual_block([high, low], 32, N, alpha)



    # 3rd block
    high = layers.AveragePooling3D(2)(high)
    low = layers.AveragePooling3D(2)(low)


    high, low = _create_octconv_residual_block([high, low], 64, N, alpha)
    # concat
    high = layers.AveragePooling3D(2)(high)
    x = layers.Concatenate()([high, low])
    x = layers.Conv3D(64, 1)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    # FC
    x = layers.GlobalAveragePooling3D()(x)
    x = layers.Dense(2, activation="softmax")(x)

    model = Model(input, x)
    return model


def octave_attention_block(input, input_channels=None, output_channels=None, encoder_depth=1,alpha=0.5,N=2):
    """
    attention block
    https://arxiv.org/abs/1704.06904
    """

    p = 1
    t = 1
    r = 1
    high,low=input
    if input_channels is None:
       input_channels = high.get_shape()[-1]
    if output_channels is None:
        output_channels = input_channels

    # First Residual Block
    for i in range(p):
        high,low =  _create_octconv_residual_block([high, low], input_channels, N, alpha)

    print("Shapes ",high.get_shape(),low.get_shape())
    # Trunc Branch
    output_trunk_high,output_trunk_low = high,low
    for i in range(t):
        output_trunk_high, output_trunk_low= _create_octconv_residual_block([output_trunk_high, output_trunk_low], input_channels, N, alpha)

    print("Shapes 2", output_trunk_high.get_shape(), output_trunk_low.get_shape())
    # Soft Mask Branch
    output_soft_mask_high = layers.MaxPool3D(padding='same')(high)  # 32x32
    output_soft_mask_low= layers.MaxPool3D(padding='same')(low)

    ## encoder
    ### first down sampling
    for i in range(r):
        output_soft_mask_high,output_soft_mask_low = _create_octconv_residual_block([output_soft_mask_high,output_soft_mask_low], input_channels, N, alpha)

    print("Shapes 3", output_soft_mask_high.get_shape(), output_soft_mask_low.get_shape())

    skip_connections_high = []
    skip_connections_low = []
    for i in range(encoder_depth - 1):

        ## skip connections
        output_skip_connection_high,output_skip_connection_low =  _create_octconv_residual_block([output_soft_mask_high,output_soft_mask_low], input_channels, N, alpha)
        skip_connections_high.append(output_skip_connection_high)
        skip_connections_low.append(output_skip_connection_low)
        # print ('skip shape:', output_skip_connection.get_shape())

        ## down sampling
        output_soft_mask_high = layers.MaxPool3D(padding='same')(output_soft_mask_high)
        output_soft_mask_low = layers.MaxPool3D(padding='same')(output_soft_mask_low)
        for _ in range(r):
            output_soft_mask_high,output_soft_mask_low =  _create_octconv_residual_block([output_soft_mask_high,output_soft_mask_low], input_channels, N, alpha)

            ## decoder
    skip_connections_high = list(reversed(skip_connections_high))
    skip_connections_low = list(reversed(skip_connections_low))

    for i in range(encoder_depth - 1):
        ## upsampling
        for _ in range(r):
            output_soft_mask_high,output_soft_mask_low =_create_octconv_residual_block([output_soft_mask_high,output_soft_mask_low], input_channels, N, alpha)
        output_soft_mask_high = layers.UpSampling3D()(output_soft_mask_high)
        output_soft_mask_high = layers.UpSampling3D()(output_soft_mask_low)
        ## skip connections
        output_soft_mask_high = layers.Add()([output_soft_mask_high, skip_connections_high[i]])
        output_soft_mask_low = layers.Add()([output_soft_mask_low, skip_connections_low[i]])

    ### last upsampling
    for i in range(r):
        output_soft_mask_high,output_soft_mask_low = _create_octconv_residual_block([output_soft_mask_high,output_soft_mask_low], input_channels, N, alpha)
    output_soft_mask_high = layers.UpSampling3D()(output_soft_mask_high)
    output_soft_mask_low = layers.UpSampling3D()(output_soft_mask_low)

    ## Output
    output_soft_mask_high = layers.Conv3D(input_channels, (1, 1,1))(output_soft_mask_high)
    output_soft_mask_high = layers.Conv3D(input_channels, (1, 1,1))(output_soft_mask_high)
    output_soft_mask_high = layers.Activation('sigmoid')(output_soft_mask_high)

    output_soft_mask_low = layers.Conv3D(input_channels, (1, 1,1))(output_soft_mask_low)
    output_soft_mask_low = layers.Conv3D(input_channels, (1, 1,1))(output_soft_mask_low)
    output_soft_mask_low = layers.Activation('sigmoid')(output_soft_mask_low)

    print("Shapes 4", output_soft_mask_high.get_shape(), output_soft_mask_low.get_shape())

    # Attention: (1 + output_soft_mask) * output_trunk
    if output_soft_mask_high.get_shape()[-1] !=output_trunk_high.get_shape()[-1]:
        output_trunk_high=layers.Conv3D(output_soft_mask_high.get_shape()[-1], (1, 1, 1))(output_trunk_high)

    if output_soft_mask_low.get_shape()[-1] !=output_trunk_low.get_shape()[-1]:
        output_trunk_low=layers.Conv3D(output_soft_mask_low.get_shape()[-1], (1, 1, 1))(output_trunk_low)

    output_high = layers.Lambda(lambda x: x + 1)(output_soft_mask_high)
    output_high = layers.Multiply()([output_high, output_trunk_high])  #

    output_low = layers.Lambda(lambda x: x + 1)(output_soft_mask_low)
    output_low = layers.Multiply()([output_low, output_trunk_low])

    # Last Residual Block
    for i in range(p):
        output_high,output_low = _create_octconv_residual_block([output_high,output_low], input_channels, N, alpha)

    return [output_high,output_low]
