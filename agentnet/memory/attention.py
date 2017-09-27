import theano.tensor as T
from ..utils.logging import warn
from ..utils.layers import DictLayer
from lasagne import init
from lasagne.layers import DenseLayer,SliceLayer,reshape,flatten,standardize

class AttentionLayer(DictLayer):
    """
    A layer that implements basic Bahdanau-style attention. Implementation is inspired by tfnn@yandex.

    Kurzgesagt, attention lets network decide which fraction of sequence/image should it view now
    by using small one-layer block that predicts (input_element,query) -> do i want to see input_element
    for all input_elements. You can read more about it here - http://distill.pub/2016/augmented-rnns/ .

    AttentionLayer also allows you to have separate keys and values: 
    it computes logits with keys, then converts them to weights(probs) and averages _values_ with those weights.

    This layer outputs a dict with keys "attn" and "probs"
    - attn - inputs processed with attention, shape [batch_size, enc_units]
    - probs - probabilities for each activation [batch_size, seq_length]

    This layer assumes input sequence/image/video/whatever to have 1 spatial dimension (see below).
    - rnn/emb format [batch,seq_len,units] works out of the box
    - 1d convolution format [batch,units,seq_len] needs dimshuffle(conv,[0,2,1])
    - 2d convolution format [batch,units,dim1,dim2] needs two-step procedure
    - step1 = dimshuffle(conv,[0,2,3,1])
    - step2 = reshape(step1,[-1,dim1*dim2,units])
    - higher dimensionality follows the same principle as 2d example above
    - reshape and dimshuffle can both be found in lasagne.layers (aliases to ReshapeLayer and DimshuffleLayer)

    When calling get_output, you can pass flag hard_attention=True to replace attention with argmax over logits.

    :param input_sequence: sequence of inputs to be processed with attention
    :type input_sequence: lasagne.layers.Layer with shape [batch,seq_length,units]

    :param query: single time-step state of decoder (usually lstm/gru/rnn hid)
    :type query: lasagne.layers.Layer with shape [batch,units]

    :param num_units: number of hidden units in attention intermediate activation
    :type num_units: int
    
    :param key_sequence: a sequence of keys to compute dot_product with. By default, uses input_sequence instead.
    :type key_sequence: lasagne.layers.Layer with shape [batch,seq_length,units] or None

    :param nonlinearity: nonlinearity in attention intermediate activation
    :type nonlinearity: function(x) -> x that works with theano tensors

    :param probs_nonlinearity: nonlinearity that converts logits of shape [batch,seq_length] into attention weights of same shape
        (you can provide softmax with tunable temperature or gumbel-softmax or anything of the sort)
    :type probs_nonlinearity: function(x) -> x that works with theano tensors


    :param mask_input: mask for input_sequence (like other lasagne masks). Default is no mask
    :type mask_input: lasagne.layers.Layer with shape [batch,seq_length]

    Other params can be theano shared variable, expression, numpy array or callable.
    Initial value, expression or initializer for the weights.
    These should be a matrix with shape ``(num_inputs, num_units)``.
    See :func:`lasagne.utils.create_param` for more information.

    The roles of those params are:
    W_enc - weights from encoder (each state) to hidden layer
    W_dec - weights from decoder (each state) to hidden layer
    W_out - hidden to logit weights
    No logit biases are introduces because softmax is invariant to adding bias to each logit

    """

    def __init__(self,
                 input_sequence,
                 query,
                 num_units,
                 mask_input = None,
                 key_sequence=None,
                 nonlinearity = T.tanh,
                 probs_nonlinearity=T.nnet.softmax,
                 W_enc = init.Normal(),
                 W_dec = init.Normal(),
                 W_out = init.Normal(),
                 **kwargs
                 ):
        assert len(input_sequence.output_shape)==3,"input_sequence must be a 3-dimensional (batch,time,units)"
        assert len(query.output_shape) == 2, "query must be a 2-dimensional for single tick (batch,units)"
        assert mask_input is None or len(mask_input.output_shape)==2,"mask_input must be 2-dimensional (batch,time) or None"
        assert key_sequence is None or len(key_sequence.output_shape) == 3, "key_sequence must be 3-dimensional " \
                                                                            "of shape (batch,time,heads,units) or None"

        #if no key sequence is given, use input_sequence as key
        key_sequence = key_sequence or input_sequence

        batch_size,seq_len,key_units = key_sequence.output_shape
        value_units = input_sequence.output_shape[-1]
        dec_units = query.output_shape[-1]


        incomings = [input_sequence, key_sequence, query]
        if mask_input is not None:
            incomings.append(mask_input)

        output_shapes = {'attn':(batch_size,value_units),
                         'probs':(batch_size,seq_len)}

        super(AttentionLayer,self).__init__(incomings,output_shapes,**kwargs)

        self.W_enc = self.add_param(W_enc,(key_units,num_units),name='enc_to_hid')
        self.W_dec = self.add_param(W_dec,(dec_units,num_units),name='dec_to_hid')
        self.W_out = self.add_param(W_out,(num_units,1),name='hid_to_logit')
        self.nonlinearity = nonlinearity
        self.probs_nonlinearity = probs_nonlinearity

    def get_output_for(self, inputs, hard_attention=False , **kwargs):
        """
        :param inputs: should consist of (enc_seq, dec) or  (enc_seq, dec, inp_mask)
        Shapes are
        enc_seq: [batch_size, seq_length, enc_units]
        dec: [batch_size, dec_units]
        inp_mask: [batch_size,seq_length] if any

        ---------------------------------
        :returns: dict with keys "attn" and "probs"
        - attn - inputs processed with attention, shape [batch_size, enc_size]
        - probs - probabilities for each activation [batch_size, ninp]
        """
        assert len(inputs) in (3,4),"inputs should be (enc_seq, dec) or  (enc_seq, dec, inp_mask)"
        mask_provided = len(inputs)==4

        #parse inputs
        enc_seq, key_seq, dec = inputs[:3]
        if mask_provided:
            mask = inputs[-1]

        #Hidden layer activations, shape [batch,seq_len,hid_units]
        hid = self.nonlinearity(
            key_seq.dot(self.W_enc) +\
            dec.dot(self.W_dec)[:,None,:]
        )


        #Logits from hidden. Mask implementation from tfnn
        logits = hid.dot(self.W_out)[:,:,0] # [batch_size,seq_len]

        if mask_provided:                  # substract large number from mask=0 time-steps
            logits -= (1 - mask) * 1000    # (written to match tfnn implementation)

        if not hard_attention:
            #regular soft attention, use softmax
            probs = self.probs_nonlinearity(logits)       # [batch_size,seq_len]

            # Compose attention.
            attn = T.sum(probs[:,:,None] * enc_seq, axis=1)

            return {'attn':attn, 'probs':probs}

        else: #hard_attention

            #use argmax over logits
            max_i = logits.argmax(axis=-1)
            batch_size = enc_seq.shape[0]
            attn = enc_seq[T.arange(batch_size),max_i]

            # one-hot probabilities
            one_hot = T.extra_ops.to_one_hot(max_i,logits.shape[1])

            return {'attn': attn, 'probs': one_hot }


class DotAttentionLayer(DictLayer):
    """
    A layer that implements multiplicative (Dotproduct) attention. Implementation is inspired by tfnn@yandex.
    
    Unlike AttentionLayer, DotAttention requires you to provide query in the same shape as one 
    time-step of the input sequence. Otherwise it does so via DenseLayer.
    
    DotAttention also allows you to have separate keys and values: 
    it computes logits with keys, then converts them to weights(probs) and averages _values_ with those weights.

    Kurzgesagt, attention lets network decide which fraction of sequence/image should it view now
    by using small one-layer block that predicts (input_element,query) -> do i want to see input_element
    for all input_elements. You can read more about it here - http://distill.pub/2016/augmented-rnns/ .

    This layer outputs a dict with keys "attn" and "probs"
    - attn - inputs processed with attention, shape [batch_size, enc_units]
    - probs - probabilities for each activation [batch_size, seq_length]

    This layer assumes input sequence/image/video/whatever to have 1 spatial dimension (see below).
    - rnn/emb format [batch,seq_len,units] works out of the box
    - 1d convolution format [batch,units,seq_len] needs dimshuffle(conv,[0,2,1])
    - 2d convolution format [batch,units,dim1,dim2] needs two-step procedure
      - step1 = dimshuffle(conv,[0,2,3,1])
      - step2 = reshape(step1,[-1,dim1*dim2,units])
    - higher dimensionality follows the same principle as 2d example above
    - reshape and dimshuffle can both be found in lasagne.layers (aliases to ReshapeLayer and DimshuffleLayer)

    When calling get_output, you can pass flag hard_attention=True to replace attention with argmax over logits.

    :param input_sequence: sequence of inputs to be processed with attention
    :type input_sequence: lasagne.layers.Layer with shape [batch,seq_length,units]

    :param query: single time-step state of decoder that is used as query (usually custom layer or lstm/gru/rnn hid)  
        If it matches input_sequence one-step size, query is used as is. 
        Otherwise, DotAttention is performed from DenseLayer(query,input_units,nonlinearity=None). 
    :type query: lasagne.layers.Layer with shape [batch,units]
    
    :param key_sequence: a sequence of keys to compute dot_product with. By default, uses input_sequence instead.
    :type key_sequence: lasagne.layers.Layer with shape [batch,seq_length,units] or None

    :param mask_input: mask for input_sequence (like other lasagne masks). Default is no mask
    :type mask_input: lasagne.layers.Layer with shape [batch,seq_length]

    :param scale: if True, scales query.dot(key) by key_size**-0.5 to maintain variance. Otherwise does nothing.
    :param use_dense_layer: if True, forcibly creates intermediate dense layer on top of query
    
    :param probs_nonlinearity: nonlinearity that converts logits of shape [batch,seq_length] into attention weights of same shape
        (you can provide softmax with tunable temperature or gumbel-softmax or anything of the sort)
    :type probs_nonlinearity: function(x) -> x that works with theano tensors
    

    """

    def __init__(self,
                 input_sequence,
                 query,
                 key_sequence = None,
                 mask_input = None,
                 scale=False,
                 use_dense_layer=False,
                 probs_nonlinearity=T.nnet.softmax,
                 **kwargs
                 ):
        assert len(input_sequence.output_shape)==3,"input_sequence must be a 3-dimensional (batch,time,units)"
        assert len(query.output_shape) == 2, "query must be a 2-dimensional for single tick (batch,units)"
        assert mask_input is None or len(mask_input.output_shape)==2,"mask_input must be 2-dimensional (batch,time) or None"
        assert key_sequence is None or len(key_sequence.output_shape) == 3, "key_sequence must be 3-dimensional " \
                                                                            "of shape (batch,time,units) or None"

        #if no key is given, use values as keys
        key_sequence = key_sequence or input_sequence

        batch_size,seq_len,key_units = key_sequence.output_shape
        value_units = input_sequence.output_shape[-1]
        dec_units = query.output_shape[-1]

        if (dec_units != key_units) and not use_dense_layer:
            warn("Input sequence and query have different unit sizes. "
                 "Using DenseLayer from query instead of raw query."
                 "To suppress this warning, set use_dense_layer=True.")
            use_dense_layer=True
        if use_dense_layer:
            name = kwargs.get('name','dotattn')
            dense_name = name+"_dense"
            query = DenseLayer(query, key_units,
                               nonlinearity=None, name=dense_name)
            dec_units = query.output_shape[-1]


        incomings = [input_sequence, key_sequence, query]
        if mask_input is not None:
            incomings.append(mask_input)

        output_shapes = {'attn':(batch_size,value_units),
                         'probs':(batch_size,seq_len)}

        super(DotAttentionLayer,self).__init__(incomings,output_shapes,**kwargs)
        self.probs_nonlinearity = probs_nonlinearity
        self.scale = scale
        self.key_size = key_units

    def get_output_for(self, inputs, hard_attention=False , **kwargs):
        """
        :param inputs: should consist of (value_seq, query) or  (value_seq, query, inp_mask)
        Shapes are
        value_seq: [batch_size, seq_length, enc_units]
        query: [batch_size, dec_units]
        inp_mask: [batch_size,seq_length] if any

        ---------------------------------
        :returns: dict with keys "attn" and "probs"
        - attn - inputs processed with attention, shape [batch_size, enc_size]
        - probs - probabilities for each activation [batch_size, ninp]
        """
        assert len(inputs) in (3,4),"inputs should be (value_seq,key_seq, query) or  (value_seq,key_seq, query, inp_mask)"
        mask_provided = len(inputs)==4

        #parse inputs
        value_seq, key_seq, query = inputs[:3]
        if mask_provided:
            mask = inputs[-1]

        #Logits from hidden. Mask implementation from tfnn

        # batch,1,units x batch,time,units, summed over units
        logits = T.sum(query[:, None, :] * key_seq, axis = -1)  #batch,time

        if self.scale:
            logits *= T.constant(self.key_size**-0.5,dtype='float32')

        if mask_provided:                  # substract large number from mask=0 time-steps
            logits -= (1 - mask) * 1000    # (written to match tfnn implementation)

        if not hard_attention:
            #regular soft attention, use softmax
            probs = self.probs_nonlinearity(logits)       # [batch_size,seq_len]

            # Compose attention.
            attn = T.sum(probs[:,:,None] * value_seq, axis=1)

            return {'attn':attn, 'probs':probs}

        else: #hard_attention

            #use argmax over logits
            max_i = logits.argmax(axis=-1)
            batch_size = value_seq.shape[0]
            attn = value_seq[T.arange(batch_size),max_i]

            # one-hot probabilities
            one_hot = T.extra_ops.to_one_hot(max_i,logits.shape[1])

            return {'attn': attn, 'probs': one_hot }


from agentnet.utils import BroadcastLayer, UnbroadcastLayer, UpcastLayer

def multihead_attention(input_sequence, query,
                        key_sequence=None, mask_input=None,
                        num_heads=1,key_size=None,value_size=None,
                        attn_class=DotAttentionLayer, name='multihead_attn',
                        **kwargs):
    """
    A convenience function that computes K attention "heads" in parallel and concatenates them. 
    Each "head" is based on num_heads linear transformations of input sequence, query, and keys 
    
    :param attn_class: what kind of attention layer to apply in multi-headed mode (Attention or DotAttention)
    :param num heads: the amount of parallel "heads"
    :param key_size: num units in attention query and key, defaults to key_sequence.shape[-1]
    :param value_size: num units in attention values, defaults to input_sequence.shape[-1] 
    
    :param input_sequence: sequence of inputs to be processed with attention
    :type input_sequence: lasagne.layers.Layer with shape [batch,seq_length,units]

    :param query: single time-step state of decoder that is used as query (usually custom layer or lstm/gru/rnn hid)  
        If it matches input_sequence one-step size, query is used as is. 
        Otherwise, DotAttention is performed from DenseLayer(query,input_units,nonlinearity=None). 
    :type query: lasagne.layers.Layer with shape [batch,units]
    
    :param key_sequence: a sequence of keys to compute dot_product with. By default, uses input_sequence instead.
    :type key_sequence: lasagne.layers.Layer with shape [batch,seq_length,units] or None

    :param mask_input: mask for input_sequence (like other lasagne masks). Default is no mask
    :type mask_input: lasagne.layers.Layer with shape [batch,seq_length]

    
    Heavily inspired by https://arxiv.org/abs/1706.03762 and http://bit.ly/2vsYX0R
    """
    assert len(input_sequence.output_shape) == 3, "input_sequence must be a 3-dimensional (batch,time,units)"
    assert len(query.output_shape) == 2, "query must be a 2-dimensional for single tick (batch,units)"
    assert mask_input is None or len(
        mask_input.output_shape) == 2, "mask_input must be 2-dimensional (batch,time) or None"
    assert key_sequence is None or len(key_sequence.output_shape) == 3, "key_sequence must be 3-dimensional " \
                                                                        "of shape (batch,time,units) or None"

    key_sequence = key_sequence or input_sequence
    key_size = key_size or key_sequence.output_shape[-1]
    value_size = value_size or input_sequence.output_shape[-1]

    def make_broadcasted_heads(incoming,head_size,name=None):
        ndim = len(incoming.output_shape)
        assert ndim in (2,3), "incoming must be 2-dimensional (query) or 3-dimensional (key or value)"

        heads = DenseLayer(incoming,head_size*num_heads,nonlinearity=None,
                           num_leading_axes=ndim-1,name=name)    #[batch,time,head_size*num_heads]

        if ndim == 3:
            heads = reshape(heads,([0],[1],head_size,num_heads), name=name)    #[batch,time,head_size,num_heads]
            broadcasted_heads = BroadcastLayer(heads, (0, 3), name=name)         #[batch*heads,time,head_size]

        else: #ndim == 2
            heads = reshape(heads, ([0], head_size, num_heads), name=name)  # [batch,head_size,num_heads]
            broadcasted_heads = BroadcastLayer(heads, (0, 2), name=name)    # [batch*heads, head_size]

        return broadcasted_heads

    query_heads = make_broadcasted_heads(query, key_size,name=name + "_query_heads")

    value_heads = make_broadcasted_heads(input_sequence, value_size, name=name + "_value_heads")

    if key_sequence is not None:
        key_heads = make_broadcasted_heads(key_sequence, key_size, name=name + "_key_heads")
    else:
        key_heads = None

    if mask_input is not None:
        mask_heads  = UpcastLayer(mask_input,broadcast_layer=query_heads)
    else:
        mask_heads = None

    attn_heads  = attn_class(value_heads,query_heads,key_sequence=key_heads,
                             mask_input=mask_heads,name=name,**kwargs)  #[batch*heads,value_size]

    attn_vectors = UnbroadcastLayer(attn_heads['attn'],broadcast_layer=query_heads) #[batch,value,heads]

    attn_vectors = flatten(attn_vectors,outdim=2)

    attn_probs = reshape(attn_heads['probs'],(-1,num_heads,[1]))   #[batch,head,probs]

    return {'attn':  attn_vectors,  #[batch, value*heads]
            'probs': attn_probs}


def self_attention(incoming, key_size=None,value_size=None,mask_input=None,name='attn',
                   attn_class=DotAttentionLayer,**kwargs):
    """
    A convenience function that applies attention from sequential layer to itself.
    
             /-> queries  -------v
    incoming --> keys     ---> attention_probs ---v
             \-> values   -------------------> attention response
             
    :param incoming: input sequence of shape [batch, time, units]
    :param key_size: num units in attention query and key, defaults to incoming.shape[-1]
    :param value_size: num units in attention values, defaults to key_size 
    :param attn_class: either DotAttentionLayer or AttentionLayer or similar layer (incl. multihead attention)
    :param kwargs: also accepts any parameters accepted by attn_class
    
    Heavily inspired by https://arxiv.org/abs/1706.03762 and http://bit.ly/2vsYX0R
    
    """
    assert len(incoming.output_shape) == 3, "incoming layer must have shape [batch,time,unit]"
    assert mask_input is None or len(mask_input.output_shape) == 2,"if mask_input is given, it must be [batch,time]"

    key_size = key_size or incoming.output_shape[-1]
    value_size = value_size or incoming.output_shape[-1]

    qkv = DenseLayer(incoming, key_size*2 + value_size, nonlinearity=None,
                         num_leading_axes=2,name=name+'.qkv')                   #[batch,time,2*key_units+value_units]

    queries = SliceLayer(qkv, slice(0,key_size),axis=-1)
    keys    = SliceLayer(qkv, slice(key_size,2*key_size), axis=-1)
    values  = SliceLayer(qkv, slice(2*key_size,qkv.num_units), axis=-1)

    # broadcast each query to every (key,value) pair
    queries_each_tick = bcast = BroadcastLayer(queries, broadcasted_axes=(0, 1)) #[batch*time,units]

    # upcast every key and value to match the amount queries
    key_for_each_query = UpcastLayer(keys, broadcast_layer=bcast)        #[batch*time, time, units]
    value_for_each_query = UpcastLayer(values, broadcast_layer=bcast)    #[batch*time, time, value_units]

    if mask_input is not None:
        mask_input = UpcastLayer(mask_input,broadcast_layer=bcast)       #[batch*time, time]

    attn_each_tick = attn_class(value_for_each_query,
                                queries_each_tick,
                                key_sequence=key_for_each_query,
                                mask_input=mask_input,
                                name=name,**kwargs)['attn']              #[batch*time, value_units]

    attn = UnbroadcastLayer(attn_each_tick, broadcast_layer=bcast)       #[batch, time, value_units]

    return attn


