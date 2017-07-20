# -*- coding: utf-8 -*-

# Created by junfeng, saj on 10/25/16.

# logging config
import logging

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p',
                    level=logging.DEBUG)
logger = logging.getLogger(__name__)

import numpy as np

import theano
import theano.tensor as T

import lasagne
from lasagne import nonlinearities
from lasagne import init
from lasagne.layers import Layer


class AnsPointerLayer(lasagne.layers.LSTMLayer):
    """
    max_steps set to 2 for boundary model.
    max_steps set to max passage length + 1 of batched passages.
    need prepare function to carefully prepare batched data.
    Not test, maybe not work.
    """
    def __init__(self, incoming, num_units, max_steps, peepholes=False, mask_input=None, **kwargs):
        """
        initialization
        :param incoming: bidirectional mLSTM for passane
        :param num_units:
        :param max_steps: max num steps to generate answer words, can be tensor scalar variable
        :param peepholes:
        :param mask_input: passage's length mask
        :param kwargs:
        """
        super(AnsPointerLayer, self).__init__(incoming, num_units, peepholes=peepholes,
                                              precompute_input=False, mask_input=mask_input,
                                              only_return_final=False, **kwargs)
        self.max_steps = max_steps
        # initializes attention weights
        input_shape = self.input_shapes[0]
        num_inputs = np.prod(input_shape[2:])
        self.V_pointer = self.add_param(init.Normal(0.1), (num_inputs, num_units), 'V_pointer')
        # doesn't need transpose
        self.v_pointer = self.add_param(init.Normal(0.1), (num_units, 1), 'v_pointer')
        self.W_a_pointer = self.add_param(init.Normal(0.1), (num_units, num_units), 'W_a_pointer')
        self.b_a_pointer = self.add_param(init.Constant(0.), (1, num_units), 'b_a_pointer')
        self.c_pointer = self.add_param(init.Constant(0.), (1, 1), 'c_pointer')

    def get_output_shape_for(self, input_shapes):
        input_shape = input_shapes[0]
        input_shape = input_shape[:2]
        # (batch_size, max_answer_words, max_input_length)
        return input_shape[0], self.max_steps, input_shape[1]

    def get_output_for(self, inputs, **kwargs):
        """
        Compute this layer's output function given a symbolic input variable

        Parameters
        ----------
        inputs : list of theano.TensorType
            `inputs[0]` should always be the symbolic input variable.  When
            this layer has a mask input (i.e. was instantiated with
            `mask_input != None`, indicating that the lengths of sequences in
            each batch vary), `inputs` should have length 2, where `inputs[1]`
            is the `mask`.  The `mask` should be supplied as a Theano variable
            denoting whether each time step in each sequence in the batch is
            part of the sequence or not.  `mask` should be a matrix of shape
            ``(n_batch, n_time_steps)`` where ``mask[i, j] = 1`` when ``j <=
            (length of sequence i)`` and ``mask[i, j] = 0`` when ``j > (length
            of sequence i)``. When the hidden state of this layer is to be
            pre-filled (i.e. was set to a :class:`Layer` instance) `inputs`
            should have length at least 2, and `inputs[-1]` is the hidden state
            to prefill with. When the cell state of this layer is to be
            pre-filled (i.e. was set to a :class:`Layer` instance) `inputs`
            should have length at least 2, and `inputs[-1]` is the hidden state
            to prefill with. When both the cell state and the hidden state are
            being pre-filled `inputs[-2]` is the hidden state, while
            `inputs[-1]` is the cell state.

        Returns
        -------
        layer_output : theano.TensorType
            Symbolic output variable.
        """
        # Retrieve the layer input
        passage = inputs[0]
        # Retrieve the mask when it is supplied
        mask = None
        hid_init = None
        cell_init = None
        if self.mask_incoming_index > 0:
            mask = inputs[self.mask_incoming_index]
        if self.hid_init_incoming_index > 0:
            hid_init = inputs[self.hid_init_incoming_index]
        if self.cell_init_incoming_index > 0:
            cell_init = inputs[self.cell_init_incoming_index]

        # Treat all dimensions after the second as flattened feature dimensions
        if passage.ndim > 3:
            passage = T.flatten(passage, 3)
        num_batch, passage_seq_len, _ = passage.shape

        # Stack input weight matrices into a (num_inputs, 4*num_units)
        # matrix, which speeds up computation
        W_in_stacked = T.concatenate(
            [self.W_in_to_ingate, self.W_in_to_forgetgate,
             self.W_in_to_cell, self.W_in_to_outgate], axis=1)

        # Same for hidden weight matrices
        W_hid_stacked = T.concatenate(
            [self.W_hid_to_ingate, self.W_hid_to_forgetgate,
             self.W_hid_to_cell, self.W_hid_to_outgate], axis=1)

        # Stack biases into a (4*num_units) vector
        b_stacked = T.concatenate(
            [self.b_ingate, self.b_forgetgate,
             self.b_cell, self.b_outgate], axis=0)

        if self.precompute_input:
            # Because the input is given for all time steps, we can
            # precompute_input the inputs dot weight matrices before scanning.
            # W_in_stacked is (n_features, 4*num_units). input is then
            # (n_time_steps, n_batch, 4*num_units).
            passage = T.dot(passage, W_in_stacked) + b_stacked

        # At each call to scan, input_n will be (n_time_steps, 4*num_units).
        # We define a slicing function that extract the input to each LSTM gate
        def slice_w(x, n):
            return x[:, n * self.num_units:(n + 1) * self.num_units]

        # Create single recurrent computation step function
        # input_i is the i'th word of the answer
        def step(input_i, cell_previous, hid_previous, *args):
            # word-by-word attention
            mh = T.dot(hid_previous, self.W_a_pointer)
            mh += self.b_a_pointer
            # mh is (n_batch, 1, n_features)
            mh = mh.dimshuffle(0, 'x', 1)
            M = T.dot(passage, self.V_pointer) + mh
            # (n_batch, passage_seq_len, n_features)
            M = nonlinearities.tanh(M)
            # alpha is (n_batch, passage_seq_len, 1)
            alpha = T.dot(M, self.v_pointer)
            # now is (n_batch, passage_seq_len)
            alpha = T.flatten(alpha, 2)
            alpha += self.c_pointer
            # 0 after softmax is not 0, fuck, my mistake.
            # when i >= passage_seq_len, fill alpha_i to -np.inf
            # apply passage_mask to alpha
            # passage_mask is (n_batch, passage_seq_len)
            alpha = T.switch(mask, alpha, -np.inf)
            alpha = T.nnet.softmax(alpha)
            # when i >= passage_seq_len, alpha_i should be 0.
            # actually not need mask, but in case of error
            # alpha = alpha * mask
            alpha = alpha.dimshuffle(0, 1, 'x')
            weighted_passage = T.sum(passage * alpha, axis=1)
            # (n_batch, n_features)
            input_n = weighted_passage
            if not self.precompute_input:
                input_n = T.dot(input_n, W_in_stacked) + b_stacked

            # Calculate gates pre-activations and slice
            gates = input_n + T.dot(hid_previous, W_hid_stacked)

            # Clip gradients
            if self.grad_clipping:
                gates = theano.gradient.grad_clip(
                    gates, -self.grad_clipping, self.grad_clipping)

            # Extract the pre-activation gate values
            ingate = slice_w(gates, 0)
            forgetgate = slice_w(gates, 1)
            cell_input = slice_w(gates, 2)
            outgate = slice_w(gates, 3)

            if self.peepholes:
                # Compute peephole connections
                ingate += cell_previous * self.W_cell_to_ingate
                forgetgate += cell_previous * self.W_cell_to_forgetgate

            # Apply nonlinearities
            ingate = self.nonlinearity_ingate(ingate)
            forgetgate = self.nonlinearity_forgetgate(forgetgate)
            cell_input = self.nonlinearity_cell(cell_input)

            # Compute new cell value
            cell = forgetgate * cell_previous + ingate * cell_input

            if self.peepholes:
                outgate += cell * self.W_cell_to_outgate
            outgate = self.nonlinearity_outgate(outgate)

            # Compute new hidden unit activation
            hid = outgate * self.nonlinearity(cell)

            return [cell, hid, alpha]

        sequences = T.arange(self.max_steps)
        step_fun = step

        ones = T.ones((num_batch, 1))
        if not isinstance(self.cell_init, Layer):
            # Dot against a 1s vector to repeat to shape (num_batch, num_units)
            cell_init = T.dot(ones, self.cell_init)
        if not isinstance(self.hid_init, Layer):
            # Dot against a 1s vector to repeat to shape (num_batch, num_units)
            hid_init = T.dot(ones, self.hid_init)

        # The hidden-to-hidden weight matrix is always used in step
        non_seqs = [W_hid_stacked]
        # The "peephole" weight matrices are only used when self.peepholes=True
        if self.peepholes:
            non_seqs += [self.W_cell_to_ingate,
                         self.W_cell_to_forgetgate,
                         self.W_cell_to_outgate]

        # When we aren't precomputing the input outside of scan, we need to
        # provide the input weights and biases to the step function
        if not self.precompute_input:
            non_seqs += [W_in_stacked, b_stacked]

        # attention weights
        non_seqs += [self.V_pointer,
                     self.W_a_pointer,
                     self.v_pointer,
                     self.b_a_pointer,
                     self.c_pointer,
                     passage,
                     mask,
                     ]
        # Scan op iterates over first dimension of input and repeatedly
        # applies the step function
        cell_out, hid_out, alphas = theano.scan(
            fn=step_fun,
            sequences=sequences,
            outputs_info=[cell_init, hid_init, None],
            go_backwards=self.backwards,
            truncate_gradient=self.gradient_steps,
            non_sequences=non_seqs,
            strict=True)[0]

        # (max_steps, n_batch, passage_seq_len)
        alphas = alphas.dimshuffle(1, 0, 2)
        # (n_batch, max_steps, passage_seq_len)
        return alphas
