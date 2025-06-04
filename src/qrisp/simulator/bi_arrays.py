"""
********************************************************************************
* Copyright (c) 2025 the Qrisp authors
*
* This program and the accompanying materials are made available under the
* terms of the Eclipse Public License 2.0 which is available at
* http://www.eclipse.org/legal/epl-2.0.
*
* This Source Code may also be made available under the following Secondary
* Licenses when the conditions for such availability set forth in the Eclipse
* Public License, v. 2.0 are satisfied: GNU General Public License, version 2
* with the GNU Classpath Exception which is
* available at https://www.gnu.org/software/classpath/license.html.
*
* SPDX-License-Identifier: EPL-2.0 OR GPL-2.0 WITH Classpath-exception-2.0
********************************************************************************
"""

import threading
import time

import numpy as np
from scipy.sparse import (
    coo_array,
    csc_array,
    csr_array,
)

import qrisp.simulator.bi_array_helper as hlp
from qrisp.simulator.numerics_config import (
    float_thresh,
    sparsification_rate,
    cutoff_ratio,
)

try:
    # sparse_dot_mkl seems to be only faster in situations, where the shape of the
    # matrices, results in better parallelization. For many quantum circuits, this was
    # not the case so this feature is disabled until further notice.
    raise Exception
    # Install command: conda install -c conda-forge sparse_dot_mkl
    from sparse_dot_mkl import dot_product_mkl

    def sparse_matrix_mult(a, b):
        if not isinstance(a, (np.ndarray, csr_array, csc_array)):
            a = a.tocsr()
        if not isinstance(b, (np.ndarray, csc_array, csr_array)):
            b = b.tocsc()

        return dot_product_mkl(a, b)

except:
    # print("Failed to import mkl sparse matrix multiplication. Install:
    # with conda install -c conda-forge sparse_dot_mkl. Using scipy algorithm.")
    sparse_matrix_mult = lambda a, b: (a @ b).tocoo()


def sparse_matrix_mult(a, b):
    return hlp.sparse_matrix_mult(a, b)


# A quick helper function to evaluate the product of an iterable
def prod(iter):
    res = 1
    for i in range(len(iter)):
        res *= iter[i]
    return res


class BiArray:
    def v_dot(self, other):
        original_shape_self = self.shape
        orignal_shape_other = other.shape

        self.reshape((self.size,))
        other.reshape((other.size,))

        res = tensordot(self, other, (0, 0))

        self.reshape(original_shape_self)
        other.reshape(orignal_shape_other)

        return res


# The classes inside this file aim describe so called BiArrays.
# A BiArray is a nd-array, where all the dimensions is are a power of 2,
# i.e. 2, 4, 8, or 1024... This property can be used to efficiently manipulate indices
# of sparse matrices such that they can be processed quickly enough to use sparse matrix
# multiplication libraries for tensor contraction. This functionality is summarized in
# the SparseBiArray class. The DenseBiArray class is an interface to numpy arrays,
# that also use the restriction of the dimension being a power of two. For instance does
# the DenseBiArray class support constant time reshaping and axis swapping. The
# reshaping and axis swappings are "collected" and executed together once the array is
# required for contraction.

# Both classes automatically make use of potential parallelism in the code.
# This is achieved by returning arrays where the content is "under construction" by a
# certain thread. If the "under construction" array is required in a contraction, the
# finishing of the calculation process is awaited.

# The tensordot function is a numpy-esque interface for performing tensor contractions.
# This function also automatically converts between sparse and dense BiArrays once a
# specified sparsity threshold is reached.

# A third class, the DummyBiArray, is a class that provides the same methods as the
# previous two, however the contractions are not actually executed, but instead the
# amount of floating point operations is logged in a class attribute. This allows to
# quickly evaluate the performance of a contraction order without actually contracting.


multithreading_threshold = np.inf

# This SparseBiArray class provides a numpy-esque rank n tensor interface, which uses
# sparse matrix multiplication for tensor contraction. The fact that only powers of two
# are allowed for the size, allow efficient index manipulations, that permit faster
# construction and processing of said sparse matrices compared to for instance the
# rank n tensor interface of the "sparse" package.

# The idea is to store multi-dimensional sparse arrays as the coo-array of their raveled
# version i.e. we ravel the array and store the indices of the non-zero entries together
# with their data. So for instance the array
# array([[0, 2, 0, 0],
#        [3, 0, 0, 0]])
# is stored as with the non-zero indices nz_indices = [1, 4] and data = [2,3].
# If we now want to swap the axes of the array in this representation, we can do
# bit-level manipulations on the nz_indices array:
# In binary we have
# nz_indices = ["001", "100"]
# An axis swap can now be performed by manipulating the binary strings (which is
# possible rather efficiently). In this case the manipulation consists of moving the
# last two digits of the bitstring to the front. After the axis swap we have
# nz_indices_axis_swapped = ["010", "001"] = [2, 1]
# If we now unravel the array with updated shape, we get
# array([[0, 3],
#        [2, 0],
#        [0, 0],
#        [0, 0]])
# Which confirms that this bit-level operation indeed represented an axis-swap


class SparseBiArray(BiArray):
    # This attribute decides after how many contractions, there should be a check for
    # duplicates and zero entries in the sparse matrix data
    contraction_counter_threshold = np.inf

    # Constructor of the SparseBiArray class.
    # init_object can be
    # - a numpy array
    # - a tuple of arrays where init_object[0] is an array of non-zero indices
    #     and init_object[1] the corresponding data
    # - a sparse array in coo format
    def __init__(self, init_object, shape=None, contraction_counter=0):
        # This attribute tracks how many sparse matrix contractions, this object has
        # been through
        self.contraction_counter = contraction_counter

        # Handle the case of an array
        if isinstance(init_object, np.ndarray):
            array = init_object

            # Store shape
            self.shape = array.shape

            # Ravel init array
            raveled_array = array.ravel()

            # raveled_array = np.around(raveled_array, 6, out = raveled_array)

            # Find non-zero indices
            self.nz_indices = np.nonzero(raveled_array)[0]

            # Store data
            self.data = raveled_array[self.nz_indices]

            # self.nz_indices, self.data = hlp.elim_zeros(self.nz_indices, self.data)

        # Handle the case that the init_object is a tuple of nz_indices/data arrays
        elif isinstance(init_object, tuple):
            if shape is None:
                raise Exception(
                    "Tried to initialise SparseBiArray from sparse data without "
                    "providing a shape"
                )

            self.nz_indices = init_object[0].ravel()
            self.data = init_object[1]

            # self.nz_indices, self.data = hlp.elim_zeros(self.nz_indices, self.data)

        # Handle the case that the init_object is a sparse matrix/array
        elif isinstance(init_object, (coo_array)):
            if shape is None:
                raise Exception(
                    "Tried to initialise SparseBiArray from sparse matrix without "
                    "providing a shape"
                )

            self.nz_indices = hlp.gen_flat_coords(
                init_object.col, init_object.row, init_object.shape
            )
            self.data = init_object.data

        elif isinstance(init_object, SparseBiArray):
            copy = init_object.copy()

            for var in copy.__dict__.keys():
                self.__dict__[var] = copy.__dict__[var]

            return
        else:
            raise Exception(
                "Tried to initialize SparseBiArray with unknown type "
                + str(type(init_object))
            )
        if shape is not None:
            self.shape = shape

        # Store the size
        self.size = prod(self.shape)

        if not ((self.size & (self.size - 1) == 0) and self.size != 0):
            raise Exception(
                "Tried to initialize SparseBiArray with size "
                + str(self.size)
                + " which is not a power of 2"
            )

        # Store the sparsity
        self.sparsity = self.nz_indices.shape[0] / self.size

        # This list keeps track of the axes permutations that have been applied.
        # Once the SparseBiArray is needed for a contraction, all these swaps are
        # applied within a single function call and then the corresponding matrix is
        # build from that.
        try:
            log_size = int(np.log2(self.size))
        except:
            size = int(self.size)
            log_size = 0
            while not size % 2:
                log_size += 1
                size = size >> 1
        self.index_bit_permutation = list(range(log_size))

        self.dtype = self.data.dtype

        # If the contraction counter is above the threshold, we sum the duplicates and
        # eliminate the zeros.
        # if False:
        # if contraction_counter > self.contraction_counter_threshold:
        # self.sum_duplicates()
        # self.eliminate_zeros()
        # self.contraction_counter = 0

    # Method for copying
    def copy(self):
        self.apply_swaps()
        return SparseBiArray(
            (self.nz_indices.copy(), self.data.copy()), shape=tuple(self.shape)
        )

    # This method checks if the sparse array has a thread object, which means that
    # it's data is under construction. If so, it waits until the thread is finished
    def catch_up(self):
        if hasattr(self, "thread"):
            start_time = time.time()
            self.thread.join()

            duration = time.time() - start_time
            if duration > 0.001:
                # print(duration)
                pass
            del self.thread

    # Reshaping method - note that no data is manipulated, i.e. constant time
    def reshape(self, shape):
        if isinstance(shape, int):
            shape = (shape,)

        shape = tuple(shape)
        if prod(shape) != prod(self.shape):
            raise Exception("Tried to reshape with invalid shape")

        self.shape = shape

        return self

    # Method for swapping two axes
    def swapaxes(self, i, j):
        if i == j:
            return
        if i > j:
            i, j = j, i

        # In order to store the swap in the index_bit_permutation attribute,
        # we need to swap the corresponding digit ranges inside this list

        # For this we first determine the digit range of the index bitstrings
        # For instance imagine we have the index 215 and the shape (4,8,8)
        # and we want to swap the axes i = 0 and j = 2.
        # In binary, 215 is 11010111.
        # We first determine the layout of the indices inside this bitstring
        bit_shape = np.log2(self.shape).astype(np.int64)

        # In our case the bit shape is (2,3,3)

        i_interval_start = np.sum(bit_shape[:i])
        i_interval_end = i_interval_start + bit_shape[i]

        # In our example i = 0, the i_interval starts at the first (most significant)
        # digit and ends after the third, i.e. |11|010111

        j_interval_start = np.sum(bit_shape[:j])
        j_interval_end = j_interval_start + bit_shape[j]

        # The j interval now starts at 6-th digit and ends at the last digit
        # ie. |11|010|111|

        # these two intervals have to be swapped.
        # However, in order to make this method constant in time, the actual swapping
        # of the indices is not performed here - rather we log the permutation of the
        # indices.

        # After creating of our SparseBiArray the attribute .index_bit_permutation
        # simply holds the numbers from 0 to log2(size)-1, i.e. in the case of our
        # example [0,1,2,3,4,5,6,7]
        # In order to log the axis swap, we perform this permutation on the
        # .index_bit_permutation attribute, i.e. after the application of this method,
        # we want it to be [5,6,7,2,3,4,0,1]
        # i.e. the interval of the first two and the last three swapped

        # In order to do this we split the index_bit_permutation list into 5 parts
        # 1. the interval before the i-interval
        # 2. the i-interval
        # 3. the interval between i and j
        # 4. the j interval
        # 5. the interval after j

        # in the case of our example, we would have
        # 1. = []
        # 2. = [0,1]
        # 3. = [2,3,4]
        # 4. = [5,6,7]
        # 5. = []

        # We than acquire the permuted list by concatenating these list in the correct
        # order, i.e. 1. + 4. + 3. + 2. + 5.

        # Determine the first interval
        init_interval = self.index_bit_permutation[:i_interval_start]

        # Determine the i-interval
        i_interval = self.index_bit_permutation[i_interval_start:i_interval_end]

        # Determine the middle-interval
        middle_interval = self.index_bit_permutation[i_interval_end:j_interval_start]

        # Determine the j-interval
        j_interval = self.index_bit_permutation[j_interval_start:j_interval_end]

        # Determine the final interval
        final_interval = self.index_bit_permutation[j_interval_end:]

        # Concatenate intervals
        self.index_bit_permutation = (
            init_interval + j_interval + middle_interval + i_interval + final_interval
        )

        # Create the tuple for the new shape
        new_shape = list(self.shape)

        new_shape[i], new_shape[j] = new_shape[j], new_shape[i]

        self.shape = tuple(new_shape)

        return self

    # This method applies the index permutation generated too
    def apply_swaps(self):
        # Wait for any contraction processes to finish
        self.catch_up()

        # Check if any permutations have to be performed
        try:
            log_size = int(np.log2(self.size))
        except:
            size = int(self.size)
            log_size = 0
            while not size % 2:
                log_size += 1
                size = size >> 1
        if self.index_bit_permutation == list(range(log_size)):
            return

        # Apply permutations
        # The bit level manipulations are done in the helper file
        self.nz_indices = hlp.permute_axes(
            self.nz_indices, np.array(self.index_bit_permutation)
        ).ravel()

        # Reset the index_bit_permutation attribute, to
        # indicate that no permutation has to be executed now
        self.index_bit_permutation = list(range(int(log_size)))

    # This method sums duplicates ie. if an index appears twice in nz_indices,
    # the corresponding data entries are summed
    def sum_duplicates(self):
        self.nz_indices, self.data = hlp.sum_duplicates(self.nz_indices, self.data)

    # If after a contraction a new zero appeared, this entry does not need to tracked
    # anymore. This method removes the zeros
    def elim_zeros(self):
        self.nz_indices, self.data = hlp.elim_zeros(self.nz_indices, self.data)

    # Builds up the sparse matrix in coo format
    # The coo format saves an array of row/column indices and the corresponding entries
    # Shape is the shape the sparse matrix should have
    def build_sr_matrix(self, shape, transpose=False):
        if prod(shape) != self.size:
            raise Exception("Tried to build sparse matrix with invalid shape")

        # Perform potential swaps
        self.apply_swaps()

        # Retrieve coordinates
        row, col = hlp.get_coordinates(self.nz_indices, shape)

        # Create sparse matrix
        if not transpose:
            res = coo_array((self.data, (row, col)), shape=shape)
        else:
            res = coo_array((self.data, (col, row)), shape=shape[::-1])

        # This attribute seems to be False which triggers a sorting algorithm,
        # which is in many cases among the most resource costly operations during
        # simulation. Setting this to True had no influence on the results tested so
        # far.
        res.has_canonical_format = True

        return res

    # This method allows moving an index to the first position without
    # changing the order of the remaining indices
    def move_index_to_front(self, index):
        # if the index is already at the front, we don't need to do anything
        if index == 0:
            return

        # The logic for creating the ne self.index_bit_permutation attribute is similar
        # as in swapaxes

        bit_shape = [int(np.log2(s)) for s in self.shape]

        new_shape = list(self.shape)
        new_shape.insert(0, new_shape.pop(index))

        self.shape = tuple(new_shape)

        index_range_start = sum(bit_shape[:index])
        index_range_end = index_range_start + bit_shape[index]

        init_perm = self.index_bit_permutation[:index_range_start]
        index_perm = self.index_bit_permutation[index_range_start:index_range_end]
        final_perm = self.index_bit_permutation[index_range_end:]
        self.index_bit_permutation = index_perm + init_perm + final_perm

    # This function performs tensor contraction along the specified axes.
    # Axes can be contracted if they have the same dimension
    # Imagine we have two tensors with shapes (2,4,64,8,16) and (8,32,4,2)
    # then we could for instance contract the following axes
    # 0 with 3
    # 1 with 2
    # 3 with 0
    # From this we will then get a tensor, which has the shape of the
    # first concatenated with the second but with the contraction indices removed
    # ie. (64,16,32)
    # The idea of this function is to move all the contraction indices to the front
    # reshape to matrices, transpose the first matrix, and then call some matrix
    # multiplication algorithm. In the case of the example the shapes after
    # moving to the front look like
    # (2,4,8,64,16) and (2,4,8,32)
    # After reshaping to matrices the shapes look like
    # (2*4*8, 64*16) and (2*4*8, 32)
    # Transpose the first matrix
    # (64*16, 2*4*8) and (2*4*8, 32)
    # Call matrix multiplication to get a matrix of shape
    # (64*16,32)
    # Finally, reshape the result
    # (64,16,32)
    def contract(self, other, axes_self, axes_other):
        # Check if the contraction size of both tensors is allowed
        contraction_size = prod([self.shape[i] for i in axes_self])
        contraction_size_other = prod([other.shape[i] for i in axes_other])

        if contraction_size != contraction_size_other:
            raise Exception(
                "Tried to contract tensor with differently sized contraction indices"
            )

        # Copy axes list in order to prevent modification
        axes_self = list(axes_self)
        axes_other = list(axes_other)

        # Move axes to the front
        for i in range(len(axes_self)):
            self.move_index_to_front(axes_self[i])

            # Note that we need to increase the index of the axes
            # greater than the axes we just moved, because
            # the insertion at the front moved the all the indices
            # before axes_self[i]
            for j in range(i + 1, len(axes_self)):
                if axes_self[i] > axes_self[j]:
                    axes_self[j] += 1

        # Perform the same for the other tensor
        for i in range(len(axes_other)):
            other.move_index_to_front(axes_other[i])

            for j in range(i + 1, len(axes_other)):
                if axes_other[i] > axes_other[j]:
                    axes_other[j] += 1

        # Determine new shape (now that the contraction axes are at the front, we can
        # simply slice the shapes)
        res_shape = list(self.shape[len(axes_self) :]) + list(
            other.shape[len(axes_other) :]
        )

        # In order to feature multiprocessing, we create an "empty" result array, which
        # will be returned after the calculation has been started. If the content is
        # needed before the calculation finished, the catch_up method will make sure the
        # calculation finishes before further processing
        res = SparseBiArray(
            (np.zeros(1, dtype=np.int64), np.zeros(1, dtype=self.data.dtype)),
            shape=res_shape,
            contraction_counter=self.contraction_counter + other.contraction_counter,
        )

        # Define multi processing wrapper
        def mp_wrapper():
            # Build up sparse matrices
            sr_matrix_self = self.build_sr_matrix(
                (contraction_size, self.size // contraction_size), transpose=True
            )
            sr_matrix_other = other.build_sr_matrix(
                (contraction_size, other.size // contraction_size)
            )

            # Perform sparse matrix multiplication

            res_sr_matrix = sparse_matrix_mult(sr_matrix_self, sr_matrix_other)

            # Acquire flattened coordinates from the helper function
            res.nz_indices = hlp.gen_flat_coords(
                res_sr_matrix.col, res_sr_matrix.row, res_sr_matrix.shape
            )

            # Set the data
            res.data = res_sr_matrix.data

            # res.nz_indices, res.data = hlp.elim_zeros(res.nz_indices, res.data)

            # Set the sparsity
            res.sparsity = len(res.nz_indices) / res.size

            # Perform consolidation operations
            if False:
                # if res.contraction_counter > self.contraction_counter_threshold:
                res.sum_duplicates()
                res.eliminate_zeros()
                res.contraction_counter = 0

        # Start the thead
        if res.size > multithreading_threshold:
            res.thread = threading.Thread(target=mp_wrapper)
            res.thread.start()
        else:
            mp_wrapper()

        return res

    # This method returns self as a numpy array
    def to_array(self):
        # Apply all swapping operations
        self.apply_swaps()

        # Construct the flat array
        if self.data.dtype == np.dtype("O"):
            res = hlp.construct_flat_array_no_jit(self.nz_indices, self.data, self.size)
        else:
            res = hlp.construct_flat_array(self.nz_indices, self.data, self.size)

        # Reshape
        return res.reshape(self.shape)

    def ravel(self):
        return self.reshape(self.size)

    def __repr__(self):
        return str(self.to_array())

    # This function allows to split the array into it's lower and upper half
    # i.e. if this SparseBiArray represents the array [0,1,2,3,4,5,6,7]
    # we return [0,1,2,3] and [4,5,6,7] as SparseBiArrays
    # This feature is important when measuring QuantumState as the norms of the
    # lower and upper half of the array determine the probability of measuring a 1 or 0
    def split(self):
        # Check if self is in a suitable shape (multi-dimensional arrays can not be
        # split as intended)
        if len(self.shape) != 1:
            raise Exception("Tried to split multi dimensional sparse array")

        # Applya swaps
        self.apply_swaps()

        # Find the boolean mask of indizes that are less than half of the size
        mask = self.nz_indices < self.size // 2

        # Acquire the lower half nz_indices
        lower_half_nz_indices = self.nz_indices[mask].copy()
        # Acquire the lower half data
        lower_half_data = self.data[mask].copy()

        # Create lower half result
        lower_half = SparseBiArray(
            (lower_half_nz_indices, lower_half_data), shape=(self.shape[0] // 2,)
        )

        # Perform the same process for the upper half but with inverted mask
        mask = np.invert(mask)
        upper_half_nz_indices = self.nz_indices[mask] - self.size // 2
        upper_half_data = self.data[mask].copy()

        upper_half = SparseBiArray(
            (upper_half_nz_indices, upper_half_data), shape=(self.shape[0] // 2,)
        )

        # Return result
        return lower_half, upper_half

    def multi_measure(self, indices, return_new_arrays=True):

        # print(np.log2(self.size))
        # sprs = self.build_sr_matrix(
        #     [2 ** (len(indices)), self.size // 2 ** len(indices)]
        # ).tocsr()

        sprs = self.build_sr_matrix(
            [2 ** (len(indices)), self.size // 2 ** len(indices)]
        )

        sprs.row, sprs.col, sprs.data = hlp.sort_indices_jitted(
            sprs.row, sprs.col, sprs.data, sprs.shape[1]
        )
        unique_markers = hlp.find_unique_markers(sprs.row)

        p_list = []
        outcome_index_list = []
        new_bi_arrays = []

        for i in range(len(unique_markers) - 1):
            temp_data = sprs.data[unique_markers[i] : unique_markers[i + 1]]

            p = np.abs(np.vdot(temp_data, temp_data))

            if p < float_thresh:
                continue

            p_list.append(p)
            outcome_index_list.append(sprs.row[unique_markers[i]])

            if return_new_arrays:

                new_bi_array = SparseBiArray(
                    (sprs.col[unique_markers[i] : unique_markers[i + 1]], temp_data),
                    shape=(self.size // 2 ** len(indices),),
                )

                new_bi_arrays.append(new_bi_array)

            else:
                new_bi_arrays.append(None)

        return new_bi_arrays, p_list, outcome_index_list

    # Should return a cheap guess whether two inputs are linearly independent
    def exclude_linear_indpendence(self, other):

        if not 0.5 < len(self.data) / len(other.data) < 2:
            return False
        return True

    # Calculate the squared norm, ie. the sesquilinear scalar product of self with
    # itself
    def squared_norm(self):
        return np.abs(np.vdot(self.data, self.data))

    def vdot(self, other):
        self.apply_swaps()
        other.apply_swaps()

        sparse_array_self = self.build_sr_matrix(shape=(1, self.size))
        sparse_array_other = other.build_sr_matrix(shape=(other.size, 1))

        sparse_array_self.data = np.conjugate(sparse_array_self.data)

        return sparse_matrix_mult(sparse_array_self, sparse_array_other).todense()[0]

        return sparse_array_self.conjugate().dot(sparse_array_other).todense()[0]

        # return sparse_array_self.conjugate().dot(sparse_array_other).todense()[0]

    # Return self as a DenseBiArray
    def to_dense(self):
        res = DenseBiArray(np.zeros(1), sparsity=self.sparsity, shape=self.shape)

        # We perform the same technique to enable multithreading as in contract,
        # ie. return an unfinished result that is waited on if required
        def mp_wrapper():
            res.data = self.to_array()

        # Start the thead
        if self.size > multithreading_threshold:
            res.thread = threading.Thread(target=mp_wrapper)
            res.thread.start()
        else:
            mp_wrapper()

        return res

    def to_sparse(self):
        return self


# This class serves mainly as an interface to numpy arrays using the same methods as the
# SparseBiArray class this way the algorithm using these classes doesn't need to care
# about wether it's treating a sparse array or a dense array.
# Furthermore, some multithreading techniques from the SparseBiArray class are deployed.
class DenseBiArray(BiArray):
    # The constructor can only be called with numpy arrays
    def __init__(self, array, sparsity=None, shape=None):
        if not isinstance(array, np.ndarray):
            raise Exception(
                "Tried to initialize DenseBiArray with type " + str(type(array))
            )

        # Save data
        self.data = array

        # Set shape
        if shape:
            self.shape = shape
        else:
            self.shape = array.shape

        # Determine size
        self.size = prod(self.shape)

        # If not given, estimate sparsity. In many situations the sparsity
        # of a contraction result can be estimated (and then given to the constructor),
        # so it is helpfull to track it
        if sparsity is None:
            self.sparsity = np.count_nonzero(array) / array.size
        else:
            # self.sparsity = np.count_nonzero(array)/array.size
            self.sparsity = sparsity

        # This attribute serves a similar purpose as in the SparseBiArray class
        self.index_bit_permutation = list(range(int(np.log2(self.size))))

        self.dtype = self.data.dtype

    # Copy method
    def copy(self):
        self.apply_swaps()
        return DenseBiArray(self.data.copy(), sparsity=self.sparsity)

    # Catch up method (same purpose as in SparseBiArray)
    def catch_up(self):
        if hasattr(self, "thread"):
            start_time = time.time()
            self.thread.join()

            duration = time.time() - start_time
            if duration > 0.001:
                # print("Waited ", duration)
                pass
            del self.thread

    # The shape of the DenseBiArray is tracked only as a tuple.
    # Once the data is required for a contraction, the actual data will be reshaped
    def reshape(self, shape):
        if isinstance(shape, int):
            shape = (shape,)

        if prod(shape) != prod(self.shape):
            raise Exception("Tried to reshape with invalid shape")

        self.shape = tuple(shape)
        return self

    # This method works similarly as it's equivalent in SparseBiArray
    def swapaxes(self, i, j):
        if i == j:
            return
        if i > j:
            i, j = j, i

        bit_shape = np.log2(self.shape).astype(np.int64)

        i_interval_start = sum(bit_shape[:i])
        i_interval_end = i_interval_start + bit_shape[i]

        j_interval_start = sum(bit_shape[:j])
        j_interval_end = j_interval_start + bit_shape[j]

        init_interval = self.index_bit_permutation[:i_interval_start]
        i_interval = self.index_bit_permutation[i_interval_start:i_interval_end]

        middle_interval = self.index_bit_permutation[i_interval_end:j_interval_start]

        j_interval = self.index_bit_permutation[j_interval_start:j_interval_end]

        final_interval = self.index_bit_permutation[j_interval_end:]

        self.index_bit_permutation = (
            init_interval + j_interval + middle_interval + i_interval + final_interval
        )

        new_shape = list(self.shape)

        new_shape[i], new_shape[j] = new_shape[j], new_shape[i]

        self.shape = tuple(new_shape)

        return self

    # This method works similarly as it's equivalent in SparseBiArray
    def move_index_to_front(self, index):
        if index == 0:
            return

        bit_shape = [int(np.log2(s)) for s in self.shape]

        new_shape = list(self.shape)
        new_shape.insert(0, new_shape.pop(index))

        self.shape = tuple(new_shape)

        index_range_start = sum(bit_shape[:index])
        index_range_end = index_range_start + bit_shape[index]

        init_perm = self.index_bit_permutation[:index_range_start]
        index_perm = self.index_bit_permutation[index_range_start:index_range_end]
        final_perm = self.index_bit_permutation[index_range_end:]
        self.index_bit_permutation = index_perm + init_perm + final_perm

    # This method works similarly as it's equivalent in SparseBiArray
    def apply_swaps(self):
        # Wait for any computation processes to finish
        self.catch_up()

        # Check if axes need to be swapped
        if self.index_bit_permutation == list(range(int(np.log2(self.size)))):
            # Check if data needs to be reshaped
            if self.data.shape != self.shape:
                self.data = self.data.reshape(self.shape)

            return

        # The plan is now to reshape into "tensor format" ie. log2(self.size) axes of
        # imension two and permute the axes

        # Numpy however does not allow arrays with more than 32 axes, so it would not be
        # possible to manipulate arrays describing more than 32 qubits.
        # In order to reduce the number of axes that need to be permuted,
        # we identify intervals of indices that do not need permutation on them
        # and group them into one axis

        # Get the permutation of the axes
        bi_index_perm = [
            self.index_bit_permutation.index(i) for i in range(int(np.log2(self.size)))
        ]

        if False:
            # result = np.empty(self.data.size, dtype = self.data.dtype)

            index_perm = hlp.invert_permutation(np.array(bi_index_perm))[::-1]
            index_perm = np.abs(len(index_perm) - 1 - index_perm)

            # flattened index array
            f_index_array = np.empty(2 ** len(index_perm), dtype=np.int64)

            f_index_array[0] = 0
            # result[0] = self.data.ravel()[0]

            # Unfortunately, this is not faster (maybe if implemented in C?)
            self.data = hlp.bi_array_moveaxis(
                self.data.ravel(), index_perm, f_index_array
            )
            # temp = hlp.bi_array_moveaxis(self.data.ravel(), index_perm, f_index_array)

            # permutated_indices = hlp.bi_array_moveaxis_jitted(index_perm, f_index_array)
            # self.data = self.data.ravel()[temp.ravel()]

            self.data.shape = self.shape

        else:
            perm = list(bi_index_perm)
            # Initialize the shape
            perm_shape = len(self.index_bit_permutation) * [2]

            i = 1
            while i < len(perm):
                # If this condition is met, both indices form an interval,
                # that does not need further permutation
                if perm[i] - 1 == perm[i - 1]:
                    # Remove the index from the perm and perm_shape list
                    perm.pop(i)
                    perm_shape.pop(i)

                    # Update the shape
                    perm_shape[i - 1] *= 2

                    # Update the permutation indices
                    for j in range(len(perm)):
                        if perm[j] > perm[i - 1]:
                            perm[j] -= 1
                else:
                    i += 1

            # Reshape data to the required shape
            self.data = self.data.reshape(perm_shape)

            # Move the axes on the data
            self.data = np.moveaxis(self.data, list(range(len(perm))), perm)

            # Reshape the data
            self.data = self.data.reshape(self.shape)

        # Set the index_bit_permutation to identity
        self.index_bit_permutation = list(range(int(np.log2(self.size))))

    # Raveling method
    def ravel(self):
        self.reshape(self.size)
        self.apply_swaps()
        return self

    # This function works similar as it's equivalent in SparseBiArray
    # i.e. identify contraction axes, move to the front, transpose one matrix
    # and then apply matrix multiplication
    def contract(self, other, axes_self, axes_other):
        # Check if the contraction size of both tensors is allowed
        contraction_size = prod([self.shape[i] for i in axes_self])
        contraction_size_other = prod([other.shape[i] for i in axes_other])

        if contraction_size != contraction_size_other:
            raise Exception(
                "Tried to contract tensor with differently sized contraction indices"
            )

        # Copy axes list in order to prevent modification
        axes_self = list(axes_self)
        axes_other = list(axes_other)

        # Move axes to the front
        for i in range(len(axes_self)):
            self.move_index_to_front(axes_self[i])

            # Note that we need to increase the index of the axes
            # greater than the axes we just moved, because
            # the insertion at the front moved the all the indices
            # before axes_self[i]
            for j in range(i + 1, len(axes_self)):
                if axes_self[i] > axes_self[j]:
                    axes_self[j] += 1

        # Perform the same for the other tensor
        for i in range(len(axes_other)):
            other.move_index_to_front(axes_other[i])

            for j in range(i + 1, len(axes_other)):
                if axes_other[i] > axes_other[j]:
                    axes_other[j] += 1

        # Determine new shape (now that the contraction axes are at the front, we can
        # simply slice the shapes).
        res_shape = list(self.shape[len(axes_self) :]) + list(
            other.shape[len(axes_other) :]
        )

        # The sparsity can be estimated using statistical methods:
        # In a regular tensor contraction, any entry of the resulting tensor, is the sum
        # of n = contraction_size products of entries of the two source tensors.
        # Each of these products is non-zero with probability
        # p_nz = self.sparsity*other.sparsity

        # p_nz = self.sparsity*other.sparsity

        # The probabilty for the product being zero is therefore
        # p_z = 1 - self.sparsity*other.sparsity

        # p_z = 1 - self.sparsity*other.sparsity

        # Therefore, we perform n = contraction_size random experiments with probability
        # p_z. The result entry is zero if all products are zero, so

        # p_res_z = p_z**contraction_size

        # To get the sparsity, we finally need the probability of the result entry being
        # non-zero.

        # res_sparsity = 1-p_res_z

        # We tested this formula and it works extremely well.

        res_sparsity = 1 - (1 - self.sparsity * other.sparsity) ** contraction_size

        if self.dtype == np.dtype("O") or other.dtype == np.dtype("O"):
            dtype = np.dtype("O")
        else:
            dtype = self.dtype

        # Prepare "unfinished" result
        res = DenseBiArray(
            np.array([self.data.ravel()[0] * other.data.ravel()[0]], dtype=dtype),
            sparsity=res_sparsity,
            shape=res_shape,
        )

        # Prepare multithreading wrapper
        # Performs the same logic as it's SparseBiArray equivalent
        def mp_wrapper():
            original_shape_self = self.shape
            original_shape_other = other.shape

            self.reshape([contraction_size, self.size // contraction_size])
            other.reshape([contraction_size, other.size // contraction_size])

            self.swapaxes(0, 1)

            self.apply_swaps()
            other.apply_swaps()

            res.data = np.matmul(self.data, other.data).ravel()

            self.swapaxes(0, 1)
            self.reshape(original_shape_self)
            other.reshape(original_shape_other)

            if np.random.random(1)[0] < sparsification_rate and res.size > 2**14:
                temp = np.abs(res.data.ravel())
                max_abs = np.max(temp)
                filter_arr = temp > max_abs * cutoff_ratio
                res.data = res.data * filter_arr
                res.data = res.data.reshape(res_shape)
                res.sparsity = np.sum(filter_arr) / res.size

        if res.size > multithreading_threshold:
            # Start the wrapper
            res.thread = threading.Thread(target=mp_wrapper)
            res.thread.start()
        else:
            mp_wrapper()
        # Return the result
        return res

    # This method works similarly as it's equivalent in SparseBiArray
    def split(self):
        if len(self.shape) != 1:
            raise Exception("Tried to split multi rank BiArray")
        self.apply_swaps()

        return DenseBiArray(
            self.data[: self.size // 2].copy(), self.sparsity
        ), DenseBiArray(self.data[self.size // 2 :].copy(), sparsity=self.sparsity)

    def multi_measure(self, indices, return_new_arrays=True):
        original_shape = tuple(self.shape)
        # self.reshape([2 ** (len(indices)), self.size // 2 ** len(indices)])
        self.reshape(self.size)

        np_array = self.to_array()

        if len(indices) > 10:
            new_arrays, p_list, outcome_index_list = hlp.dense_measurement_brute(
                np_array, len(indices), 0
            )
        else:
            new_arrays, p_list, outcome_index_list = hlp.dense_measurement_smart(
                np_array, len(indices), 0
            )

        new_bi_arrays = []

        if return_new_arrays:
            for i in range(len(new_arrays)):
                new_bi_arrays.append(
                    DenseBiArray(new_arrays[i], sparsity=self.sparsity)
                )
        else:
            new_bi_arrays = len(p_list) * [None]

        self.reshape(original_shape)

        return new_bi_arrays, p_list, outcome_index_list

    # This method works similarly as it's equivalent in SparseBiArray
    def squared_norm(self):
        self.apply_swaps()
        return np.abs(np.vdot(self.data, self.data))

    def vdot(self, other):
        self.apply_swaps()
        other.apply_swaps()
        return np.vdot(self.data, other.data)

    # Should return a cheap guess whether two inputs are linearly independent
    def exclude_linear_indpendence(self, other):
        return True

    # This method works similarly as it's equivalent in SparseBiArray
    def to_array(self):
        self.apply_swaps()
        return self.data

    # This method works similarly as it's equivalent in SparseBiArray
    def to_dense(self):
        return self

    # Generates the SparseBiArray version of self using multithreading
    def to_sparse(self):
        res = SparseBiArray(
            (np.zeros(1, dtype=np.int64), np.zeros(1)), shape=self.shape
        )

        # Prepare the wrapper
        def mp_wrapper():
            self.apply_swaps()

            # Ravel init array
            raveled_array = self.data.ravel()

            # Find non-zero indices
            res.nz_indices = np.nonzero(raveled_array)[0]

            # Store data
            res.data = raveled_array[res.nz_indices]

            res.nz_indices, res.data = hlp.elim_zeros(res.nz_indices, res.data)

            res.sparsity = len(res.nz_indices) / res.size

        # Start the
        if res.size > multithreading_threshold:
            res.thread = threading.Thread(target=mp_wrapper)
            res.thread.start()
        else:
            mp_wrapper()
        return res

    def __repr__(self):
        return str(self.to_array())


# This class is used to estimate the complexity of a contraction order
# The contract method does not actually execute a contraction but only logs
# the estimated amount of floating point operations
class DummyBiArray(BiArray):
    current_memory = np.zeros(1)
    max_memory = np.zeros(1)
    flops = np.zeros(1)

    def __init__(self, data=None, shape=None, sparsity=1):
        self.sparsity = sparsity
        if not isinstance(data, type(None)):
            self.shape = data.shape
            self.sparsity = np.count_nonzero(data) / data.size
        if not isinstance(shape, type(None)):
            self.shape = tuple(shape)

        self.size = prod(self.shape)

    def reshape(self, shape):
        self.shape = shape

    def swapaxes(self, axis_0, axis_1):
        self.shape = list(self.shape)
        self.shape[axis_0], self.shape[axis_1] = self.shape[axis_1], self.shape[axis_0]
        self.shape = tuple(self.shape)

    def contract(self, other, axes_self, axes_other):
        contraction_size = prod([self.shape[i] for i in axes_self])

        temp_shape_self = list(self.shape)
        temp_shape_other = list(other.shape)

        delete_multiple_element(temp_shape_self, axes_self)
        delete_multiple_element(temp_shape_other, axes_other)

        res_sparsity = 1 - (1 - self.sparsity * other.sparsity) ** contraction_size
        res_shape = temp_shape_self + temp_shape_other
        try:
            self.flops += prod(res_shape) * contraction_size
        except TypeError:
            self.flops += np.inf

        res = DummyBiArray(shape=res_shape, sparsity=res_sparsity)
        self.current_memory[0] += res.size - self.size - other.size

        if self.current_memory > self.max_memory:
            self.max_memory[0] = self.current_memory[0]

        return res

    def copy(self):
        return DummyBiArray(shape=self.shape)

    @classmethod
    def reset_stat_counter(self):
        self.current_memory = np.zeros(1)
        self.max_memory = np.zeros(1)
        self.flops = np.zeros(1)


# This function emulates the behavior of np.tensordot
def tensordot(a, b, axes, contract_sparsity_threshold=0.01):
    # Turn axes object into a list
    axes = list(axes)
    if isinstance(axes[0], int):
        axes[0] = [axes[0]]
    if isinstance(axes[1], int):
        axes[1] = [axes[1]]

    # Treat the case of a dummy contraction
    if isinstance(a, DummyBiArray) and isinstance(b, DummyBiArray):
        return a.contract(b, axes[0], axes[1])

    # Treat the case of a numpy contraction
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        return np.tensordot(a, b, axes)

    if a.data.dtype == np.dtype("O") or b.data.dtype == np.dtype("O"):
        a = a.to_dense()
        b = b.to_dense()

        return a.contract(b, axes[0], axes[1])

    if len(axes[0]) != 0:
        contraction_size = prod([a.shape[i] for i in axes[0]])

        # Estimate resulting sparsity (for more information on this estimate check the
        # contract method of DenseBiArray)
        res_sparsity = 1 - (1 - a.sparsity * b.sparsity) ** contraction_size

        res_size = a.size * b.size / contraction_size
    else:
        res_sparsity = a.sparsity * b.sparsity

        res_size = a.size * b.size

    if (
        res_sparsity > contract_sparsity_threshold or a.size * b.size < 2**12
    ) and not res_size > 2**32:
        #     if ((isinstance(a, SparseBiArray) and isinstance(b, DenseBiArray)) or
        #     (isinstance(b, SparseBiArray) and isinstance(a, DenseBiArray))):

        #         return mixed_contractor(a, b, axes)

        a = a.to_dense()
        b = b.to_dense()

    else:
        a = a.to_sparse()
        b = b.to_sparse()

    # Treat the case of a "contractionless" tensor product
    if len(axes[0]) == 0:
        # We utilize the multithreading pattern presented in the previous contraction
        # methods.
        if isinstance(a, SparseBiArray) and isinstance(b, SparseBiArray):
            # Prepare result arrays
            res = SparseBiArray(
                (np.zeros(1, dtype=np.int64), np.ones(1)),
                shape=list(a.shape) + list(b.shape),
            )

            # Prepare wrapper
            def mp_wrapper():
                # Apply any potential swaps on the source arrays
                a.apply_swaps()
                b.apply_swaps()

                # Imagine we have two arrays with flattened nz_indices [1,2,3], [2,3,6]
                # and sizes 16, 8  respectively
                # the nz_indices of the result can be ordered in a matrix
                # [1,2,3] + 2*16
                # [1,2,3] + 3*16
                # [1,2,3] + 6*16
                # (where the plus is executed on all entries)

                # This idea is encapsulated in the following command
                nz_indices = np.tensordot(
                    b.size * a.nz_indices,
                    np.ones(len(b.data), dtype=np.int64),
                    ((), ()),
                ) + np.tensordot(
                    np.ones(len(a.data), dtype=np.int64), b.nz_indices, ((), ())
                )

                # The corresponding data can be calculated as
                data = np.tensordot(a.data.ravel(), b.data.ravel(), ((), ())).ravel()

                # Set nz_indices and data
                res.nz_indices = nz_indices.ravel()

                res.data = data

            if res.size > multithreading_threshold:
                res.thread = threading.Thread(target=mp_wrapper)
                res.thread.start()
            else:
                mp_wrapper()

            return res

        else:
            # For the DenseBiArray, we can simply use the numpy tensordot
            a.catch_up()
            b.catch_up()

            res = DenseBiArray(
                np.array([a.data.ravel()[0] * b.data.ravel()[0]]),
                sparsity=a.sparsity * b.sparsity,
                shape=list(a.shape) + list(b.shape),
            )

            def mp_wrapper():
                a.apply_swaps()
                b.apply_swaps()

                res.data = np.tensordot(a.to_array(), b.to_array(), axes=((), ()))

            if res.size > multithreading_threshold:
                res.thread = threading.Thread(target=mp_wrapper)
                res.thread.start()
            else:
                mp_wrapper()

            return res

    return a.contract(b, axes[0], axes[1])


def mixed_contractor(a, b, axes):
    if not (
        (isinstance(a, SparseBiArray) and isinstance(b, DenseBiArray))
        or (isinstance(b, SparseBiArray) and isinstance(a, DenseBiArray))
    ):
        raise Exception("Tried to call mixed contraction on non-mixed input")

    # Copy axes list in order to prevent modification
    axes_a = list(axes[0])
    axes_b = list(axes[1])

    # Check if the contraction size of both tensors is allowed
    contraction_size = prod([a.shape[i] for i in axes_a])
    contraction_size_b = prod([b.shape[i] for i in axes_b])

    if contraction_size != contraction_size_b:
        raise Exception(
            "Tried to contract tensor with differently sized contraction indices"
        )

    # Move axes to the front
    for i in range(len(axes_a)):
        a.move_index_to_front(axes_a[i])

        # Note that we need to increase the index of the axes
        # greater than the axes we just moved, because
        # the insertion at the front moved the all the indices
        # before axes_a[i]
        for j in range(i + 1, len(axes_a)):
            if axes_a[i] > axes_a[j]:
                axes_a[j] += 1

    # Perform the same for the b tensor
    for i in range(len(axes_b)):
        b.move_index_to_front(axes_b[i])

        for j in range(i + 1, len(axes_b)):
            if axes_b[i] > axes_b[j]:
                axes_b[j] += 1

    # Determine new shape (now that the contraction axes are at the front, we can simply
    # slice the shapes)
    res_shape = list(a.shape[len(axes_a) :]) + list(b.shape[len(axes_b) :])

    res_sparsity = 1 - (1 - a.sparsity * b.sparsity) ** contraction_size

    # Prepare "unfinished" result
    res = DenseBiArray(
        np.empty(1, dtype=a.data.dtype), sparsity=res_sparsity, shape=res_shape
    )

    # Define multi processing wrapper
    def mp_wrapper():
        # Build up sparse matrices
        if isinstance(a, SparseBiArray):
            mult_matrix_a = a.build_sr_matrix(
                (contraction_size, a.size // contraction_size), transpose=True
            )
            original_shape_b = b.shape
            b.reshape([contraction_size, b.size // contraction_size])
            b.apply_swaps()
            mult_matrix_b = b.data

        else:
            original_shape_a = a.shape
            a.reshape([contraction_size, a.size // contraction_size])
            a.swapaxes(0, 1)
            a.apply_swaps()
            mult_matrix_a = a.data
            mult_matrix_b = b.build_sr_matrix(
                (contraction_size, b.size // contraction_size)
            )

        # Set the data
        res.data = sparse_matrix_mult(mult_matrix_a, mult_matrix_b)

    # Start the thead
    if res.size > multithreading_threshold:
        res.thread = threading.Thread(target=mp_wrapper)
        res.thread.start()
    else:
        mp_wrapper()

    return res


def delete_multiple_element(list_object, indices):
    indices = sorted(indices, reverse=True)
    for idx in indices:
        if idx < len(list_object):
            list_object.pop(idx)
