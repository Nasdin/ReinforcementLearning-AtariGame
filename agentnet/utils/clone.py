"""
Utility functions that can clone lasagne network layers in a custom way.
[Will be] used for:
- target networks, e.g. older copies of NN used for reference Qvalues.
- DPG-like methods where critic has to process both optimal and actual actions

"""
from copy import deepcopy

import lasagne

from .format import check_list, check_ordered_dict
from ..utils.logging import warn


from .layers import reapply as _reapply
def reapply(*args,**kwargs):
    warn("DEPRECATION: Reapply has been moved to agentnet.utils.layers.reapply (or agentnet.utils.reapply)."
         "It will be removed from agentnet.utils.clone in the next release.")
    return _reapply(*args,**kwargs)

def clone_network(original_network, bottom_layers=None,
                  share_params=False, share_inputs=True,name_prefix = None):
    """
    Creates a copy of lasagne network layer(s) provided as original_network.

    If bottom_layers is a list of layers or a single layer, function won't
    copy these layers, using existing ones instead.

    Else, if bottom_layers is a dictionary of {existing_layer:new_layer},
    each time original network would have used existing_layer, cloned network uses new_layer

    It is possible to either use existing weights or clone them via share_weights flag.
    If weights are shared, target_network will always have same weights as original one.
    Any changes (e.g. loading or training) will affect both original and cloned network.
    This is useful if you want both networks to train together (i.e. you have same network applied twice)
    One example of such case is Deep DPG algorithm: http://arxiv.org/abs/1509.02971

    Otherwise, if weights are NOT shared, the cloned network will begin with same weights as
    the original one at the moment it was cloned, but than the two networks will be completely independent.
    This is useful if you want cloned network to deviate from original. One example is when you
    need a "target network" for your deep RL agent, that stores older weights snapshot.
    The DQN that uses this trick can be found here: https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf


    :param original_network: A network to be cloned (all output layers)
    :type original_network: lasagne.layers.Layer or list/tuple/dict/any_iterable of such.
        If list, layers must be VALUES, not keys.
    :param bottom_layers: the layers which you don't want to clone. See description above.
        This parameter can also contain ARBITRARY objects within the original_network that you want to share.
    :type bottom_layers: lasagne.layers.Layer or a list/tuple/dict of such.
    :param share_params: if True, cloned network will use same shared variables for weights.
        Otherwise new shared variables will be created and set to original NN values.
        WARNING! shared weights must be accessible via lasagne.layers.get_all_params with no flags
        If you want custom other parameters to be shared, use bottom_layers
    :param share_inputs: if True, all InputLayers will still be shared even if not mentioned in bottom_layers
    :type share_inputs: bool
    :param name_prefix: if not None, adds this prefix to all the layers and params of the cloned network
    :type name_prefix: string or None
    :return: a clone of original_network (whether layer, list, dict, tuple or whatever
    """

    if isinstance(original_network,dict):
        original_layers = check_ordered_dict(original_network).values()
    else:#original_layers is a layer or a list/tuple of such
        original_layers = check_list(original_network)

    #filling memo: a dictionary of {id -> stored_item} items that should NOT be copied, but instead reused.
    memo = {}

    if bottom_layers is None:
        #in this case, copy the entire network
        pass
    if isinstance(bottom_layers,dict):
        #make a substitude: each time copier meets original layer, it replaces that layer with custom replacement
        for original,replacement in bottom_layers.items():
            memo[id(original)] = replacement

    else: #case type(bottom_layers) in [lasagne.layers.Layer,list,tuple]
        #make sure inputs are kept the same
        bottom_layers = check_list(bottom_layers)
        for layer in bottom_layers:
            memo[id(layer)] = layer

    #add shared weights
    if share_params:
        warn("clone_network with share_params=True may be unreliable in some cases. "\
             "If you want to simply apply the network elsewhere, use agentnet.utils.layers.reapply")
        all_weights = lasagne.layers.get_all_params(original_layers)
        for weight_var in all_weights:
            #if weight already in memo
            if id(weight_var) in memo:
                #variable is shared if replacement id matches memo key id. Otherwise it's "replaced"
                existing_item = memo[id(weight_var)]
                status = "shared" if id(existing_item) == id(weight_var) else "replaced with {}".format(existing_item)
                warn("Param {} was already {} manually. Default sharing because of share_params was redundant.".format(
                    weight_var, status
                ))
            else:
                #no collisions in memo. Simply add new unit
                memo[id(weight_var)] = weight_var

    #add shared InputLayers
    if share_inputs:
        all_layers = lasagne.layers.get_all_layers(original_layers)
        input_layers = filter(lambda l: isinstance(l,lasagne.layers.InputLayer), all_layers)

        for l_inp in input_layers:
            # if layer already in memo
            if id(l_inp) in memo:
                # layer is shared if replacement id matches memo key id. Otherwise it's "replaced"
                existing_item = memo[id(l_inp)]
                status = "shared" if id(existing_item) == id(l_inp) else "replaced with {}".format(existing_item)
                warn("Layer {} was already {} manually. Default sharing because of share_inputs was redundant.".format(
                    l_inp, status))
            else:
                # no collisions in memo. Simply add new unit
                memo[id(l_inp)] = l_inp

    network_clone = deepcopy(original_network,memo=memo)

    #substitute names, if asked
    if name_prefix is not None:
        #get list of clone output layers
        if isinstance(network_clone, dict):
            clone_layers = check_ordered_dict(network_clone).values()
        else:  # original_layers is a layer or a list/tuple of such
            clone_layers = check_list(network_clone)

        #substitute layer names
        all_original_layers = set(lasagne.layers.get_all_layers(original_layers))
        all_clone_layers = lasagne.layers.get_all_layers(clone_layers)

        for layer in all_clone_layers:
            if layer not in all_original_layers:
                #substitute cloned layer name
                layer.name = name_prefix + (layer.name or '')
            #otherwise it's a shared layer

        #substitute param names
        all_original_params = set(lasagne.layers.get_all_params(original_layers))
        all_clone_params = lasagne.layers.get_all_params(clone_layers)

        for param in all_clone_params:
            if param not in all_original_params:
                # substitute cloned param name
                param.name = name_prefix + (param.name or '')
            # otherwise it's a shared param


    return network_clone



