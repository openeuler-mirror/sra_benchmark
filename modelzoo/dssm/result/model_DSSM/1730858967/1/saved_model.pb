´'
,ä+
.
Abs
x"T
y"T"
Ttype:

2	
:
Add
x"T
y"T
z"T"
Ttype:
2	
A
AddV2
x"T
y"T
z"T"
Ttype:
2	
x
Assign
ref"T

value"T

output_ref"T"	
Ttype"
validate_shapebool("
use_lockingbool(
s
	AssignSub
ref"T

value"T

output_ref"T" 
Ttype:
2	"
use_lockingbool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
p
GatherNd
params"Tparams
indices"Tindices
output"Tparams"
Tparamstype"
Tindicestype:
2	
­
GatherV2
params"Tparams
indices"Tindices
axis"Taxis
output"Tparams"

batch_dimsint "
Tparamstype"
Tindicestype:
2	"
Taxistype:
2	
B
GreaterEqual
x"T
y"T
z
"
Ttype:
2	
V
HistogramSummary
tag
values"T
summary"
Ttype0:
2	
.
Identity

input"T
output"T"	
Ttype
?
	LessEqual
x"T
y"T
z
"
Ttype:
2	
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
8
Maximum
x"T
y"T
z"T"
Ttype:

2	

Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
N
Merge
inputs"T*N
output"T
value_index"	
Ttype"
Nint(0
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
=
Mul
x"T
y"T
z"T"
Ttype:
2	

NoOp
k
NotEqual
x"T
y"T
z
"
Ttype:
2	
"$
incompatible_shape_errorbool(
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
X
PlaceholderWithDefault
input"dtype
output"dtype"
dtypetype"
shapeshape

Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
~
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	
>
RealDiv
x"T
y"T
z"T"
Ttype:
2	
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
0
Round
x"T
y"T"
Ttype:

2	
.
Rsqrt
x"T
y"T"
Ttype:

2
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
P
ScalarSummary
tags
values"T
summary"
Ttype:
2	
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
O
Size

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
a
Slice

input"T
begin"Index
size"Index
output"T"	
Ttype"
Indextype:
2	
ˇ
SparseFillEmptyRows
indices	
values"T
dense_shape	
default_value"T
output_indices	
output_values"T
empty_row_indicator

reverse_index_map	"	
Ttype
h
SparseReshape
input_indices	
input_shape	
	new_shape	
output_indices	
output_shape	
z
SparseSegmentMean	
data"T
indices"Tidx
segment_ids
output"T"
Ttype:
2"
Tidxtype0:
2	
1
Square
x"T
y"T"
Ttype:

2	
G
SquaredDifference
x"T
y"T
z"T"
Ttype:

2	
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
2
StopGradient

input"T
output"T"	
Ttype
ö
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
G
StringToHashBucketFast	
input

output	"
num_bucketsint(0
:
Sub
x"T
y"T
z"T"
Ttype:
2	

Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
M
Switch	
data"T
pred

output_false"T
output_true"T"	
Ttype
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	

TruncatedNormal

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	
P
Unique
x"T
y"T
idx"out_idx"	
Ttype"
out_idxtype0:
2	
s

VariableV2
ref"dtype"
shapeshape"
dtypetype"
	containerstring "
shared_namestring 
E
Where

input"T	
index	"%
Ttype0
:
2	

&
	ZerosLike
x"T
y"T"	
Ttype"serve*1.15.52unknown8żË"

global_step/Initializer/zerosConst*
value	B	 R *
_class
loc:@global_step*
dtype0	*
_output_shapes
: 
k
global_step
VariableV2*
_output_shapes
: *
_class
loc:@global_step*
shape: *
dtype0	

global_step/AssignAssignglobal_stepglobal_step/Initializer/zeros*
T0	*
_class
loc:@global_step*
_output_shapes
: 
j
global_step/readIdentityglobal_step*
T0	*
_class
loc:@global_step*
_output_shapes
: 
f
PlaceholderPlaceholder*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*
shape:˙˙˙˙˙˙˙˙˙
h
Placeholder_1Placeholder*
dtype0*
shape:˙˙˙˙˙˙˙˙˙*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
h
Placeholder_2Placeholder*
dtype0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
shape:˙˙˙˙˙˙˙˙˙
h
Placeholder_3Placeholder*
shape:˙˙˙˙˙˙˙˙˙*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0
h
Placeholder_4Placeholder*
dtype0*
shape:˙˙˙˙˙˙˙˙˙*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
h
Placeholder_5Placeholder*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*
shape:˙˙˙˙˙˙˙˙˙
h
Placeholder_6Placeholder*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
shape:˙˙˙˙˙˙˙˙˙*
dtype0
h
Placeholder_7Placeholder*
dtype0*
shape:˙˙˙˙˙˙˙˙˙*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
h
Placeholder_8Placeholder*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*
shape:˙˙˙˙˙˙˙˙˙
h
Placeholder_9Placeholder*
dtype0*
shape:˙˙˙˙˙˙˙˙˙*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
i
Placeholder_10Placeholder*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*
shape:˙˙˙˙˙˙˙˙˙
i
Placeholder_11Placeholder*
dtype0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
shape:˙˙˙˙˙˙˙˙˙
i
Placeholder_12Placeholder*
dtype0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
shape:˙˙˙˙˙˙˙˙˙
i
Placeholder_13Placeholder*
dtype0*
shape:˙˙˙˙˙˙˙˙˙*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
i
Placeholder_14Placeholder*
shape:˙˙˙˙˙˙˙˙˙*
dtype0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
i
Placeholder_15Placeholder*
dtype0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
shape:˙˙˙˙˙˙˙˙˙
i
Placeholder_16Placeholder*
dtype0*
shape:˙˙˙˙˙˙˙˙˙*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
i
Placeholder_17Placeholder*
shape:˙˙˙˙˙˙˙˙˙*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0

:input_layer/input_layer/age_level_embedding/ExpandDims/dimConst*
_output_shapes
: *
valueB :
˙˙˙˙˙˙˙˙˙*
dtype0
Â
6input_layer/input_layer/age_level_embedding/ExpandDims
ExpandDimsPlaceholder_12:input_layer/input_layer/age_level_embedding/ExpandDims/dim*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

Jinput_layer/input_layer/age_level_embedding/to_sparse_input/ignore_value/xConst*
valueB B *
_output_shapes
: *
dtype0

Dinput_layer/input_layer/age_level_embedding/to_sparse_input/NotEqualNotEqual6input_layer/input_layer/age_level_embedding/ExpandDimsJinput_layer/input_layer/age_level_embedding/to_sparse_input/ignore_value/x*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
ť
Cinput_layer/input_layer/age_level_embedding/to_sparse_input/indicesWhereDinput_layer/input_layer/age_level_embedding/to_sparse_input/NotEqual*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

Binput_layer/input_layer/age_level_embedding/to_sparse_input/valuesGatherNd6input_layer/input_layer/age_level_embedding/ExpandDimsCinput_layer/input_layer/age_level_embedding/to_sparse_input/indices*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
Tparams0*
Tindices0	
˝
Ginput_layer/input_layer/age_level_embedding/to_sparse_input/dense_shapeShape6input_layer/input_layer/age_level_embedding/ExpandDims*
T0*
out_type0	*
_output_shapes
:
Č
2input_layer/input_layer/age_level_embedding/lookupStringToHashBucketFastBinput_layer/input_layer/age_level_embedding/to_sparse_input/values*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
num_buckets


ginput_layer/input_layer/age_level_embedding/embedding_weights/part_0/Initializer/truncated_normal/shapeConst*
dtype0*W
_classM
KIloc:@input_layer/input_layer/age_level_embedding/embedding_weights/part_0*
_output_shapes
:*
valueB"
      

finput_layer/input_layer/age_level_embedding/embedding_weights/part_0/Initializer/truncated_normal/meanConst*
valueB
 *    *W
_classM
KIloc:@input_layer/input_layer/age_level_embedding/embedding_weights/part_0*
_output_shapes
: *
dtype0

hinput_layer/input_layer/age_level_embedding/embedding_weights/part_0/Initializer/truncated_normal/stddevConst*
valueB
 *  >*
_output_shapes
: *W
_classM
KIloc:@input_layer/input_layer/age_level_embedding/embedding_weights/part_0*
dtype0
ü
qinput_layer/input_layer/age_level_embedding/embedding_weights/part_0/Initializer/truncated_normal/TruncatedNormalTruncatedNormalginput_layer/input_layer/age_level_embedding/embedding_weights/part_0/Initializer/truncated_normal/shape*
T0*
_output_shapes

:
*
dtype0*W
_classM
KIloc:@input_layer/input_layer/age_level_embedding/embedding_weights/part_0
Ë
einput_layer/input_layer/age_level_embedding/embedding_weights/part_0/Initializer/truncated_normal/mulMulqinput_layer/input_layer/age_level_embedding/embedding_weights/part_0/Initializer/truncated_normal/TruncatedNormalhinput_layer/input_layer/age_level_embedding/embedding_weights/part_0/Initializer/truncated_normal/stddev*
_output_shapes

:
*W
_classM
KIloc:@input_layer/input_layer/age_level_embedding/embedding_weights/part_0*
T0
š
ainput_layer/input_layer/age_level_embedding/embedding_weights/part_0/Initializer/truncated_normalAddeinput_layer/input_layer/age_level_embedding/embedding_weights/part_0/Initializer/truncated_normal/mulfinput_layer/input_layer/age_level_embedding/embedding_weights/part_0/Initializer/truncated_normal/mean*W
_classM
KIloc:@input_layer/input_layer/age_level_embedding/embedding_weights/part_0*
T0*
_output_shapes

:

í
Dinput_layer/input_layer/age_level_embedding/embedding_weights/part_0
VariableV2*
shape
:
*W
_classM
KIloc:@input_layer/input_layer/age_level_embedding/embedding_weights/part_0*
_output_shapes

:
*
dtype0

Kinput_layer/input_layer/age_level_embedding/embedding_weights/part_0/AssignAssignDinput_layer/input_layer/age_level_embedding/embedding_weights/part_0ainput_layer/input_layer/age_level_embedding/embedding_weights/part_0/Initializer/truncated_normal*
T0*
_output_shapes

:
*W
_classM
KIloc:@input_layer/input_layer/age_level_embedding/embedding_weights/part_0

Iinput_layer/input_layer/age_level_embedding/embedding_weights/part_0/readIdentityDinput_layer/input_layer/age_level_embedding/embedding_weights/part_0*
T0*
_output_shapes

:
*W
_classM
KIloc:@input_layer/input_layer/age_level_embedding/embedding_weights/part_0

Sinput_layer/input_layer/age_level_embedding/age_level_embedding_weights/Slice/beginConst*
valueB: *
_output_shapes
:*
dtype0

Rinput_layer/input_layer/age_level_embedding/age_level_embedding_weights/Slice/sizeConst*
_output_shapes
:*
valueB:*
dtype0
ú
Minput_layer/input_layer/age_level_embedding/age_level_embedding_weights/SliceSliceGinput_layer/input_layer/age_level_embedding/to_sparse_input/dense_shapeSinput_layer/input_layer/age_level_embedding/age_level_embedding_weights/Slice/beginRinput_layer/input_layer/age_level_embedding/age_level_embedding_weights/Slice/size*
Index0*
T0	*
_output_shapes
:

Minput_layer/input_layer/age_level_embedding/age_level_embedding_weights/ConstConst*
dtype0*
_output_shapes
:*
valueB: 

Linput_layer/input_layer/age_level_embedding/age_level_embedding_weights/ProdProdMinput_layer/input_layer/age_level_embedding/age_level_embedding_weights/SliceMinput_layer/input_layer/age_level_embedding/age_level_embedding_weights/Const*
T0	*
_output_shapes
: 

Xinput_layer/input_layer/age_level_embedding/age_level_embedding_weights/GatherV2/indicesConst*
dtype0*
_output_shapes
: *
value	B :

Uinput_layer/input_layer/age_level_embedding/age_level_embedding_weights/GatherV2/axisConst*
value	B : *
dtype0*
_output_shapes
: 

Pinput_layer/input_layer/age_level_embedding/age_level_embedding_weights/GatherV2GatherV2Ginput_layer/input_layer/age_level_embedding/to_sparse_input/dense_shapeXinput_layer/input_layer/age_level_embedding/age_level_embedding_weights/GatherV2/indicesUinput_layer/input_layer/age_level_embedding/age_level_embedding_weights/GatherV2/axis*
Tparams0	*
_output_shapes
: *
Tindices0*
Taxis0
¤
Ninput_layer/input_layer/age_level_embedding/age_level_embedding_weights/Cast/xPackLinput_layer/input_layer/age_level_embedding/age_level_embedding_weights/ProdPinput_layer/input_layer/age_level_embedding/age_level_embedding_weights/GatherV2*
T0	*
_output_shapes
:*
N
ó
Uinput_layer/input_layer/age_level_embedding/age_level_embedding_weights/SparseReshapeSparseReshapeCinput_layer/input_layer/age_level_embedding/to_sparse_input/indicesGinput_layer/input_layer/age_level_embedding/to_sparse_input/dense_shapeNinput_layer/input_layer/age_level_embedding/age_level_embedding_weights/Cast/x*-
_output_shapes
:˙˙˙˙˙˙˙˙˙:
Ě
^input_layer/input_layer/age_level_embedding/age_level_embedding_weights/SparseReshape/IdentityIdentity2input_layer/input_layer/age_level_embedding/lookup*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0	

Vinput_layer/input_layer/age_level_embedding/age_level_embedding_weights/GreaterEqual/yConst*
dtype0	*
value	B	 R *
_output_shapes
: 
Ę
Tinput_layer/input_layer/age_level_embedding/age_level_embedding_weights/GreaterEqualGreaterEqual^input_layer/input_layer/age_level_embedding/age_level_embedding_weights/SparseReshape/IdentityVinput_layer/input_layer/age_level_embedding/age_level_embedding_weights/GreaterEqual/y*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0	
Ő
Minput_layer/input_layer/age_level_embedding/age_level_embedding_weights/WhereWhereTinput_layer/input_layer/age_level_embedding/age_level_embedding_weights/GreaterEqual*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
¨
Uinput_layer/input_layer/age_level_embedding/age_level_embedding_weights/Reshape/shapeConst*
dtype0*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙
Ž
Oinput_layer/input_layer/age_level_embedding/age_level_embedding_weights/ReshapeReshapeMinput_layer/input_layer/age_level_embedding/age_level_embedding_weights/WhereUinput_layer/input_layer/age_level_embedding/age_level_embedding_weights/Reshape/shape*
T0	*#
_output_shapes
:˙˙˙˙˙˙˙˙˙

Winput_layer/input_layer/age_level_embedding/age_level_embedding_weights/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
´
Rinput_layer/input_layer/age_level_embedding/age_level_embedding_weights/GatherV2_1GatherV2Uinput_layer/input_layer/age_level_embedding/age_level_embedding_weights/SparseReshapeOinput_layer/input_layer/age_level_embedding/age_level_embedding_weights/ReshapeWinput_layer/input_layer/age_level_embedding/age_level_embedding_weights/GatherV2_1/axis*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
Tparams0	*
Taxis0*
Tindices0	

Winput_layer/input_layer/age_level_embedding/age_level_embedding_weights/GatherV2_2/axisConst*
dtype0*
value	B : *
_output_shapes
: 
š
Rinput_layer/input_layer/age_level_embedding/age_level_embedding_weights/GatherV2_2GatherV2^input_layer/input_layer/age_level_embedding/age_level_embedding_weights/SparseReshape/IdentityOinput_layer/input_layer/age_level_embedding/age_level_embedding_weights/ReshapeWinput_layer/input_layer/age_level_embedding/age_level_embedding_weights/GatherV2_2/axis*
Tparams0	*
Tindices0	*
Taxis0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ú
Pinput_layer/input_layer/age_level_embedding/age_level_embedding_weights/IdentityIdentityWinput_layer/input_layer/age_level_embedding/age_level_embedding_weights/SparseReshape:1*
_output_shapes
:*
T0	
Ł
ainput_layer/input_layer/age_level_embedding/age_level_embedding_weights/SparseFillEmptyRows/ConstConst*
value	B	 R *
_output_shapes
: *
dtype0	
Â
oinput_layer/input_layer/age_level_embedding/age_level_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowsSparseFillEmptyRowsRinput_layer/input_layer/age_level_embedding/age_level_embedding_weights/GatherV2_1Rinput_layer/input_layer/age_level_embedding/age_level_embedding_weights/GatherV2_2Pinput_layer/input_layer/age_level_embedding/age_level_embedding_weights/Identityainput_layer/input_layer/age_level_embedding/age_level_embedding_weights/SparseFillEmptyRows/Const*
T0	*T
_output_shapesB
@:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
Ä
sinput_layer/input_layer/age_level_embedding/age_level_embedding_weights/embedding_lookup_sparse/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        
Ć
uinput_layer/input_layer/age_level_embedding/age_level_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1Const*
_output_shapes
:*
valueB"       *
dtype0
Ć
uinput_layer/input_layer/age_level_embedding/age_level_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
Ć
minput_layer/input_layer/age_level_embedding/age_level_embedding_weights/embedding_lookup_sparse/strided_sliceStridedSliceoinput_layer/input_layer/age_level_embedding/age_level_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowssinput_layer/input_layer/age_level_embedding/age_level_embedding_weights/embedding_lookup_sparse/strided_slice/stackuinput_layer/input_layer/age_level_embedding/age_level_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1uinput_layer/input_layer/age_level_embedding/age_level_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2*
end_mask*

begin_mask*
Index0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0	*
shrink_axis_mask

dinput_layer/input_layer/age_level_embedding/age_level_embedding_weights/embedding_lookup_sparse/CastCastminput_layer/input_layer/age_level_embedding/age_level_embedding_weights/embedding_lookup_sparse/strided_slice*

DstT0*

SrcT0	*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
finput_layer/input_layer/age_level_embedding/age_level_embedding_weights/embedding_lookup_sparse/UniqueUniqueqinput_layer/input_layer/age_level_embedding/age_level_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0	

uinput_layer/input_layer/age_level_embedding/age_level_embedding_weights/embedding_lookup_sparse/embedding_lookup/axisConst*
value	B : *
dtype0*
_output_shapes
: *W
_classM
KIloc:@input_layer/input_layer/age_level_embedding/embedding_weights/part_0
Ô
pinput_layer/input_layer/age_level_embedding/age_level_embedding_weights/embedding_lookup_sparse/embedding_lookupGatherV2Iinput_layer/input_layer/age_level_embedding/embedding_weights/part_0/readfinput_layer/input_layer/age_level_embedding/age_level_embedding_weights/embedding_lookup_sparse/Uniqueuinput_layer/input_layer/age_level_embedding/age_level_embedding_weights/embedding_lookup_sparse/embedding_lookup/axis*
Tindices0	*
Tparams0*
Taxis0*W
_classM
KIloc:@input_layer/input_layer/age_level_embedding/embedding_weights/part_0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Š
yinput_layer/input_layer/age_level_embedding/age_level_embedding_weights/embedding_lookup_sparse/embedding_lookup/IdentityIdentitypinput_layer/input_layer/age_level_embedding/age_level_embedding_weights/embedding_lookup_sparse/embedding_lookup*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
ń
_input_layer/input_layer/age_level_embedding/age_level_embedding_weights/embedding_lookup_sparseSparseSegmentMeanyinput_layer/input_layer/age_level_embedding/age_level_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identityhinput_layer/input_layer/age_level_embedding/age_level_embedding_weights/embedding_lookup_sparse/Unique:1dinput_layer/input_layer/age_level_embedding/age_level_embedding_weights/embedding_lookup_sparse/Cast*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
¨
Winput_layer/input_layer/age_level_embedding/age_level_embedding_weights/Reshape_1/shapeConst*
dtype0*
valueB"˙˙˙˙   *
_output_shapes
:
Ú
Qinput_layer/input_layer/age_level_embedding/age_level_embedding_weights/Reshape_1Reshapeqinput_layer/input_layer/age_level_embedding/age_level_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:2Winput_layer/input_layer/age_level_embedding/age_level_embedding_weights/Reshape_1/shape*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

Ü
Minput_layer/input_layer/age_level_embedding/age_level_embedding_weights/ShapeShape_input_layer/input_layer/age_level_embedding/age_level_embedding_weights/embedding_lookup_sparse*
T0*
_output_shapes
:
Ľ
[input_layer/input_layer/age_level_embedding/age_level_embedding_weights/strided_slice/stackConst*
dtype0*
valueB:*
_output_shapes
:
§
]input_layer/input_layer/age_level_embedding/age_level_embedding_weights/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
§
]input_layer/input_layer/age_level_embedding/age_level_embedding_weights/strided_slice/stack_2Const*
valueB:*
_output_shapes
:*
dtype0

Uinput_layer/input_layer/age_level_embedding/age_level_embedding_weights/strided_sliceStridedSliceMinput_layer/input_layer/age_level_embedding/age_level_embedding_weights/Shape[input_layer/input_layer/age_level_embedding/age_level_embedding_weights/strided_slice/stack]input_layer/input_layer/age_level_embedding/age_level_embedding_weights/strided_slice/stack_1]input_layer/input_layer/age_level_embedding/age_level_embedding_weights/strided_slice/stack_2*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: 

Oinput_layer/input_layer/age_level_embedding/age_level_embedding_weights/stack/0Const*
_output_shapes
: *
value	B :*
dtype0
Ť
Minput_layer/input_layer/age_level_embedding/age_level_embedding_weights/stackPackOinput_layer/input_layer/age_level_embedding/age_level_embedding_weights/stack/0Uinput_layer/input_layer/age_level_embedding/age_level_embedding_weights/strided_slice*
N*
T0*
_output_shapes
:
ą
Linput_layer/input_layer/age_level_embedding/age_level_embedding_weights/TileTileQinput_layer/input_layer/age_level_embedding/age_level_embedding_weights/Reshape_1Minput_layer/input_layer/age_level_embedding/age_level_embedding_weights/stack*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
T0

ň
Rinput_layer/input_layer/age_level_embedding/age_level_embedding_weights/zeros_like	ZerosLike_input_layer/input_layer/age_level_embedding/age_level_embedding_weights/embedding_lookup_sparse*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

Ginput_layer/input_layer/age_level_embedding/age_level_embedding_weightsSelectLinput_layer/input_layer/age_level_embedding/age_level_embedding_weights/TileRinput_layer/input_layer/age_level_embedding/age_level_embedding_weights/zeros_like_input_layer/input_layer/age_level_embedding/age_level_embedding_weights/embedding_lookup_sparse*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ó
Ninput_layer/input_layer/age_level_embedding/age_level_embedding_weights/Cast_1CastGinput_layer/input_layer/age_level_embedding/to_sparse_input/dense_shape*

DstT0*

SrcT0	*
_output_shapes
:

Uinput_layer/input_layer/age_level_embedding/age_level_embedding_weights/Slice_1/beginConst*
dtype0*
valueB: *
_output_shapes
:

Tinput_layer/input_layer/age_level_embedding/age_level_embedding_weights/Slice_1/sizeConst*
dtype0*
valueB:*
_output_shapes
:

Oinput_layer/input_layer/age_level_embedding/age_level_embedding_weights/Slice_1SliceNinput_layer/input_layer/age_level_embedding/age_level_embedding_weights/Cast_1Uinput_layer/input_layer/age_level_embedding/age_level_embedding_weights/Slice_1/beginTinput_layer/input_layer/age_level_embedding/age_level_embedding_weights/Slice_1/size*
_output_shapes
:*
T0*
Index0
Ć
Oinput_layer/input_layer/age_level_embedding/age_level_embedding_weights/Shape_1ShapeGinput_layer/input_layer/age_level_embedding/age_level_embedding_weights*
T0*
_output_shapes
:

Uinput_layer/input_layer/age_level_embedding/age_level_embedding_weights/Slice_2/beginConst*
dtype0*
_output_shapes
:*
valueB:
§
Tinput_layer/input_layer/age_level_embedding/age_level_embedding_weights/Slice_2/sizeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
_output_shapes
:*
dtype0

Oinput_layer/input_layer/age_level_embedding/age_level_embedding_weights/Slice_2SliceOinput_layer/input_layer/age_level_embedding/age_level_embedding_weights/Shape_1Uinput_layer/input_layer/age_level_embedding/age_level_embedding_weights/Slice_2/beginTinput_layer/input_layer/age_level_embedding/age_level_embedding_weights/Slice_2/size*
_output_shapes
:*
T0*
Index0

Sinput_layer/input_layer/age_level_embedding/age_level_embedding_weights/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
˙
Ninput_layer/input_layer/age_level_embedding/age_level_embedding_weights/concatConcatV2Oinput_layer/input_layer/age_level_embedding/age_level_embedding_weights/Slice_1Oinput_layer/input_layer/age_level_embedding/age_level_embedding_weights/Slice_2Sinput_layer/input_layer/age_level_embedding/age_level_embedding_weights/concat/axis*
_output_shapes
:*
T0*
N
§
Qinput_layer/input_layer/age_level_embedding/age_level_embedding_weights/Reshape_2ReshapeGinput_layer/input_layer/age_level_embedding/age_level_embedding_weightsNinput_layer/input_layer/age_level_embedding/age_level_embedding_weights/concat*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
˛
1input_layer/input_layer/age_level_embedding/ShapeShapeQinput_layer/input_layer/age_level_embedding/age_level_embedding_weights/Reshape_2*
_output_shapes
:*
T0

?input_layer/input_layer/age_level_embedding/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:

Ainput_layer/input_layer/age_level_embedding/strided_slice/stack_1Const*
valueB:*
_output_shapes
:*
dtype0

Ainput_layer/input_layer/age_level_embedding/strided_slice/stack_2Const*
_output_shapes
:*
valueB:*
dtype0

9input_layer/input_layer/age_level_embedding/strided_sliceStridedSlice1input_layer/input_layer/age_level_embedding/Shape?input_layer/input_layer/age_level_embedding/strided_slice/stackAinput_layer/input_layer/age_level_embedding/strided_slice/stack_1Ainput_layer/input_layer/age_level_embedding/strided_slice/stack_2*
T0*
shrink_axis_mask*
_output_shapes
: *
Index0
}
;input_layer/input_layer/age_level_embedding/Reshape/shape/1Const*
dtype0*
value	B :*
_output_shapes
: 
ç
9input_layer/input_layer/age_level_embedding/Reshape/shapePack9input_layer/input_layer/age_level_embedding/strided_slice;input_layer/input_layer/age_level_embedding/Reshape/shape/1*
N*
_output_shapes
:*
T0
ţ
3input_layer/input_layer/age_level_embedding/ReshapeReshapeQinput_layer/input_layer/age_level_embedding/age_level_embedding_weights/Reshape_29input_layer/input_layer/age_level_embedding/Reshape/shape*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

=input_layer/input_layer/cms_group_id_embedding/ExpandDims/dimConst*
_output_shapes
: *
valueB :
˙˙˙˙˙˙˙˙˙*
dtype0
Č
9input_layer/input_layer/cms_group_id_embedding/ExpandDims
ExpandDimsPlaceholder_10=input_layer/input_layer/cms_group_id_embedding/ExpandDims/dim*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

Minput_layer/input_layer/cms_group_id_embedding/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB B 

Ginput_layer/input_layer/cms_group_id_embedding/to_sparse_input/NotEqualNotEqual9input_layer/input_layer/cms_group_id_embedding/ExpandDimsMinput_layer/input_layer/cms_group_id_embedding/to_sparse_input/ignore_value/x*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Á
Finput_layer/input_layer/cms_group_id_embedding/to_sparse_input/indicesWhereGinput_layer/input_layer/cms_group_id_embedding/to_sparse_input/NotEqual*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

Einput_layer/input_layer/cms_group_id_embedding/to_sparse_input/valuesGatherNd9input_layer/input_layer/cms_group_id_embedding/ExpandDimsFinput_layer/input_layer/cms_group_id_embedding/to_sparse_input/indices*
Tparams0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
Tindices0	
Ă
Jinput_layer/input_layer/cms_group_id_embedding/to_sparse_input/dense_shapeShape9input_layer/input_layer/cms_group_id_embedding/ExpandDims*
out_type0	*
T0*
_output_shapes
:
Î
5input_layer/input_layer/cms_group_id_embedding/lookupStringToHashBucketFastEinput_layer/input_layer/cms_group_id_embedding/to_sparse_input/values*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
num_bucketsd

jinput_layer/input_layer/cms_group_id_embedding/embedding_weights/part_0/Initializer/truncated_normal/shapeConst*
_output_shapes
:*
dtype0*
valueB"d      *Z
_classP
NLloc:@input_layer/input_layer/cms_group_id_embedding/embedding_weights/part_0

iinput_layer/input_layer/cms_group_id_embedding/embedding_weights/part_0/Initializer/truncated_normal/meanConst*
valueB
 *    *Z
_classP
NLloc:@input_layer/input_layer/cms_group_id_embedding/embedding_weights/part_0*
dtype0*
_output_shapes
: 

kinput_layer/input_layer/cms_group_id_embedding/embedding_weights/part_0/Initializer/truncated_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  >*Z
_classP
NLloc:@input_layer/input_layer/cms_group_id_embedding/embedding_weights/part_0

tinput_layer/input_layer/cms_group_id_embedding/embedding_weights/part_0/Initializer/truncated_normal/TruncatedNormalTruncatedNormaljinput_layer/input_layer/cms_group_id_embedding/embedding_weights/part_0/Initializer/truncated_normal/shape*Z
_classP
NLloc:@input_layer/input_layer/cms_group_id_embedding/embedding_weights/part_0*
T0*
dtype0*
_output_shapes

:d
×
hinput_layer/input_layer/cms_group_id_embedding/embedding_weights/part_0/Initializer/truncated_normal/mulMultinput_layer/input_layer/cms_group_id_embedding/embedding_weights/part_0/Initializer/truncated_normal/TruncatedNormalkinput_layer/input_layer/cms_group_id_embedding/embedding_weights/part_0/Initializer/truncated_normal/stddev*
T0*
_output_shapes

:d*Z
_classP
NLloc:@input_layer/input_layer/cms_group_id_embedding/embedding_weights/part_0
Ĺ
dinput_layer/input_layer/cms_group_id_embedding/embedding_weights/part_0/Initializer/truncated_normalAddhinput_layer/input_layer/cms_group_id_embedding/embedding_weights/part_0/Initializer/truncated_normal/muliinput_layer/input_layer/cms_group_id_embedding/embedding_weights/part_0/Initializer/truncated_normal/mean*Z
_classP
NLloc:@input_layer/input_layer/cms_group_id_embedding/embedding_weights/part_0*
T0*
_output_shapes

:d
ó
Ginput_layer/input_layer/cms_group_id_embedding/embedding_weights/part_0
VariableV2*
shape
:d*
dtype0*
_output_shapes

:d*Z
_classP
NLloc:@input_layer/input_layer/cms_group_id_embedding/embedding_weights/part_0

Ninput_layer/input_layer/cms_group_id_embedding/embedding_weights/part_0/AssignAssignGinput_layer/input_layer/cms_group_id_embedding/embedding_weights/part_0dinput_layer/input_layer/cms_group_id_embedding/embedding_weights/part_0/Initializer/truncated_normal*Z
_classP
NLloc:@input_layer/input_layer/cms_group_id_embedding/embedding_weights/part_0*
_output_shapes

:d*
T0
Ś
Linput_layer/input_layer/cms_group_id_embedding/embedding_weights/part_0/readIdentityGinput_layer/input_layer/cms_group_id_embedding/embedding_weights/part_0*Z
_classP
NLloc:@input_layer/input_layer/cms_group_id_embedding/embedding_weights/part_0*
T0*
_output_shapes

:d
Ł
Yinput_layer/input_layer/cms_group_id_embedding/cms_group_id_embedding_weights/Slice/beginConst*
_output_shapes
:*
dtype0*
valueB: 
˘
Xinput_layer/input_layer/cms_group_id_embedding/cms_group_id_embedding_weights/Slice/sizeConst*
dtype0*
_output_shapes
:*
valueB:

Sinput_layer/input_layer/cms_group_id_embedding/cms_group_id_embedding_weights/SliceSliceJinput_layer/input_layer/cms_group_id_embedding/to_sparse_input/dense_shapeYinput_layer/input_layer/cms_group_id_embedding/cms_group_id_embedding_weights/Slice/beginXinput_layer/input_layer/cms_group_id_embedding/cms_group_id_embedding_weights/Slice/size*
_output_shapes
:*
T0	*
Index0

Sinput_layer/input_layer/cms_group_id_embedding/cms_group_id_embedding_weights/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
Ľ
Rinput_layer/input_layer/cms_group_id_embedding/cms_group_id_embedding_weights/ProdProdSinput_layer/input_layer/cms_group_id_embedding/cms_group_id_embedding_weights/SliceSinput_layer/input_layer/cms_group_id_embedding/cms_group_id_embedding_weights/Const*
T0	*
_output_shapes
: 
 
^input_layer/input_layer/cms_group_id_embedding/cms_group_id_embedding_weights/GatherV2/indicesConst*
dtype0*
_output_shapes
: *
value	B :

[input_layer/input_layer/cms_group_id_embedding/cms_group_id_embedding_weights/GatherV2/axisConst*
_output_shapes
: *
value	B : *
dtype0
Ż
Vinput_layer/input_layer/cms_group_id_embedding/cms_group_id_embedding_weights/GatherV2GatherV2Jinput_layer/input_layer/cms_group_id_embedding/to_sparse_input/dense_shape^input_layer/input_layer/cms_group_id_embedding/cms_group_id_embedding_weights/GatherV2/indices[input_layer/input_layer/cms_group_id_embedding/cms_group_id_embedding_weights/GatherV2/axis*
Taxis0*
Tparams0	*
Tindices0*
_output_shapes
: 
ś
Tinput_layer/input_layer/cms_group_id_embedding/cms_group_id_embedding_weights/Cast/xPackRinput_layer/input_layer/cms_group_id_embedding/cms_group_id_embedding_weights/ProdVinput_layer/input_layer/cms_group_id_embedding/cms_group_id_embedding_weights/GatherV2*
_output_shapes
:*
T0	*
N

[input_layer/input_layer/cms_group_id_embedding/cms_group_id_embedding_weights/SparseReshapeSparseReshapeFinput_layer/input_layer/cms_group_id_embedding/to_sparse_input/indicesJinput_layer/input_layer/cms_group_id_embedding/to_sparse_input/dense_shapeTinput_layer/input_layer/cms_group_id_embedding/cms_group_id_embedding_weights/Cast/x*-
_output_shapes
:˙˙˙˙˙˙˙˙˙:
Ő
dinput_layer/input_layer/cms_group_id_embedding/cms_group_id_embedding_weights/SparseReshape/IdentityIdentity5input_layer/input_layer/cms_group_id_embedding/lookup*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0	

\input_layer/input_layer/cms_group_id_embedding/cms_group_id_embedding_weights/GreaterEqual/yConst*
dtype0	*
value	B	 R *
_output_shapes
: 
Ü
Zinput_layer/input_layer/cms_group_id_embedding/cms_group_id_embedding_weights/GreaterEqualGreaterEqualdinput_layer/input_layer/cms_group_id_embedding/cms_group_id_embedding_weights/SparseReshape/Identity\input_layer/input_layer/cms_group_id_embedding/cms_group_id_embedding_weights/GreaterEqual/y*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0	
á
Sinput_layer/input_layer/cms_group_id_embedding/cms_group_id_embedding_weights/WhereWhereZinput_layer/input_layer/cms_group_id_embedding/cms_group_id_embedding_weights/GreaterEqual*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ž
[input_layer/input_layer/cms_group_id_embedding/cms_group_id_embedding_weights/Reshape/shapeConst*
dtype0*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙
Ŕ
Uinput_layer/input_layer/cms_group_id_embedding/cms_group_id_embedding_weights/ReshapeReshapeSinput_layer/input_layer/cms_group_id_embedding/cms_group_id_embedding_weights/Where[input_layer/input_layer/cms_group_id_embedding/cms_group_id_embedding_weights/Reshape/shape*
T0	*#
_output_shapes
:˙˙˙˙˙˙˙˙˙

]input_layer/input_layer/cms_group_id_embedding/cms_group_id_embedding_weights/GatherV2_1/axisConst*
value	B : *
_output_shapes
: *
dtype0
Ě
Xinput_layer/input_layer/cms_group_id_embedding/cms_group_id_embedding_weights/GatherV2_1GatherV2[input_layer/input_layer/cms_group_id_embedding/cms_group_id_embedding_weights/SparseReshapeUinput_layer/input_layer/cms_group_id_embedding/cms_group_id_embedding_weights/Reshape]input_layer/input_layer/cms_group_id_embedding/cms_group_id_embedding_weights/GatherV2_1/axis*
Taxis0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
Tindices0	*
Tparams0	

]input_layer/input_layer/cms_group_id_embedding/cms_group_id_embedding_weights/GatherV2_2/axisConst*
dtype0*
_output_shapes
: *
value	B : 
Ń
Xinput_layer/input_layer/cms_group_id_embedding/cms_group_id_embedding_weights/GatherV2_2GatherV2dinput_layer/input_layer/cms_group_id_embedding/cms_group_id_embedding_weights/SparseReshape/IdentityUinput_layer/input_layer/cms_group_id_embedding/cms_group_id_embedding_weights/Reshape]input_layer/input_layer/cms_group_id_embedding/cms_group_id_embedding_weights/GatherV2_2/axis*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
Tindices0	*
Tparams0	*
Taxis0
ć
Vinput_layer/input_layer/cms_group_id_embedding/cms_group_id_embedding_weights/IdentityIdentity]input_layer/input_layer/cms_group_id_embedding/cms_group_id_embedding_weights/SparseReshape:1*
T0	*
_output_shapes
:
Š
ginput_layer/input_layer/cms_group_id_embedding/cms_group_id_embedding_weights/SparseFillEmptyRows/ConstConst*
value	B	 R *
dtype0	*
_output_shapes
: 
ŕ
uinput_layer/input_layer/cms_group_id_embedding/cms_group_id_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowsSparseFillEmptyRowsXinput_layer/input_layer/cms_group_id_embedding/cms_group_id_embedding_weights/GatherV2_1Xinput_layer/input_layer/cms_group_id_embedding/cms_group_id_embedding_weights/GatherV2_2Vinput_layer/input_layer/cms_group_id_embedding/cms_group_id_embedding_weights/Identityginput_layer/input_layer/cms_group_id_embedding/cms_group_id_embedding_weights/SparseFillEmptyRows/Const*
T0	*T
_output_shapesB
@:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
Ę
yinput_layer/input_layer/cms_group_id_embedding/cms_group_id_embedding_weights/embedding_lookup_sparse/strided_slice/stackConst*
valueB"        *
_output_shapes
:*
dtype0
Ě
{input_layer/input_layer/cms_group_id_embedding/cms_group_id_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1Const*
dtype0*
valueB"       *
_output_shapes
:
Ě
{input_layer/input_layer/cms_group_id_embedding/cms_group_id_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2Const*
dtype0*
valueB"      *
_output_shapes
:
ä
sinput_layer/input_layer/cms_group_id_embedding/cms_group_id_embedding_weights/embedding_lookup_sparse/strided_sliceStridedSliceuinput_layer/input_layer/cms_group_id_embedding/cms_group_id_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowsyinput_layer/input_layer/cms_group_id_embedding/cms_group_id_embedding_weights/embedding_lookup_sparse/strided_slice/stack{input_layer/input_layer/cms_group_id_embedding/cms_group_id_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1{input_layer/input_layer/cms_group_id_embedding/cms_group_id_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2*
end_mask*
T0	*

begin_mask*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
shrink_axis_mask*
Index0
¤
jinput_layer/input_layer/cms_group_id_embedding/cms_group_id_embedding_weights/embedding_lookup_sparse/CastCastsinput_layer/input_layer/cms_group_id_embedding/cms_group_id_embedding_weights/embedding_lookup_sparse/strided_slice*

SrcT0	*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*

DstT0
Ź
linput_layer/input_layer/cms_group_id_embedding/cms_group_id_embedding_weights/embedding_lookup_sparse/UniqueUniquewinput_layer/input_layer/cms_group_id_embedding/cms_group_id_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:1*
T0	*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙

{input_layer/input_layer/cms_group_id_embedding/cms_group_id_embedding_weights/embedding_lookup_sparse/embedding_lookup/axisConst*
value	B : *
dtype0*Z
_classP
NLloc:@input_layer/input_layer/cms_group_id_embedding/embedding_weights/part_0*
_output_shapes
: 
ě
vinput_layer/input_layer/cms_group_id_embedding/cms_group_id_embedding_weights/embedding_lookup_sparse/embedding_lookupGatherV2Linput_layer/input_layer/cms_group_id_embedding/embedding_weights/part_0/readlinput_layer/input_layer/cms_group_id_embedding/cms_group_id_embedding_weights/embedding_lookup_sparse/Unique{input_layer/input_layer/cms_group_id_embedding/cms_group_id_embedding_weights/embedding_lookup_sparse/embedding_lookup/axis*Z
_classP
NLloc:@input_layer/input_layer/cms_group_id_embedding/embedding_weights/part_0*
Tindices0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
Taxis0*
Tparams0
ľ
input_layer/input_layer/cms_group_id_embedding/cms_group_id_embedding_weights/embedding_lookup_sparse/embedding_lookup/IdentityIdentityvinput_layer/input_layer/cms_group_id_embedding/cms_group_id_embedding_weights/embedding_lookup_sparse/embedding_lookup*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

einput_layer/input_layer/cms_group_id_embedding/cms_group_id_embedding_weights/embedding_lookup_sparseSparseSegmentMeaninput_layer/input_layer/cms_group_id_embedding/cms_group_id_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identityninput_layer/input_layer/cms_group_id_embedding/cms_group_id_embedding_weights/embedding_lookup_sparse/Unique:1jinput_layer/input_layer/cms_group_id_embedding/cms_group_id_embedding_weights/embedding_lookup_sparse/Cast*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ž
]input_layer/input_layer/cms_group_id_embedding/cms_group_id_embedding_weights/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   
ě
Winput_layer/input_layer/cms_group_id_embedding/cms_group_id_embedding_weights/Reshape_1Reshapewinput_layer/input_layer/cms_group_id_embedding/cms_group_id_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:2]input_layer/input_layer/cms_group_id_embedding/cms_group_id_embedding_weights/Reshape_1/shape*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

č
Sinput_layer/input_layer/cms_group_id_embedding/cms_group_id_embedding_weights/ShapeShapeeinput_layer/input_layer/cms_group_id_embedding/cms_group_id_embedding_weights/embedding_lookup_sparse*
_output_shapes
:*
T0
Ť
ainput_layer/input_layer/cms_group_id_embedding/cms_group_id_embedding_weights/strided_slice/stackConst*
dtype0*
valueB:*
_output_shapes
:
­
cinput_layer/input_layer/cms_group_id_embedding/cms_group_id_embedding_weights/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
­
cinput_layer/input_layer/cms_group_id_embedding/cms_group_id_embedding_weights/strided_slice/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
ł
[input_layer/input_layer/cms_group_id_embedding/cms_group_id_embedding_weights/strided_sliceStridedSliceSinput_layer/input_layer/cms_group_id_embedding/cms_group_id_embedding_weights/Shapeainput_layer/input_layer/cms_group_id_embedding/cms_group_id_embedding_weights/strided_slice/stackcinput_layer/input_layer/cms_group_id_embedding/cms_group_id_embedding_weights/strided_slice/stack_1cinput_layer/input_layer/cms_group_id_embedding/cms_group_id_embedding_weights/strided_slice/stack_2*
Index0*
T0*
shrink_axis_mask*
_output_shapes
: 

Uinput_layer/input_layer/cms_group_id_embedding/cms_group_id_embedding_weights/stack/0Const*
_output_shapes
: *
dtype0*
value	B :
˝
Sinput_layer/input_layer/cms_group_id_embedding/cms_group_id_embedding_weights/stackPackUinput_layer/input_layer/cms_group_id_embedding/cms_group_id_embedding_weights/stack/0[input_layer/input_layer/cms_group_id_embedding/cms_group_id_embedding_weights/strided_slice*
_output_shapes
:*
N*
T0
Ă
Rinput_layer/input_layer/cms_group_id_embedding/cms_group_id_embedding_weights/TileTileWinput_layer/input_layer/cms_group_id_embedding/cms_group_id_embedding_weights/Reshape_1Sinput_layer/input_layer/cms_group_id_embedding/cms_group_id_embedding_weights/stack*
T0
*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
ţ
Xinput_layer/input_layer/cms_group_id_embedding/cms_group_id_embedding_weights/zeros_like	ZerosLikeeinput_layer/input_layer/cms_group_id_embedding/cms_group_id_embedding_weights/embedding_lookup_sparse*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

Minput_layer/input_layer/cms_group_id_embedding/cms_group_id_embedding_weightsSelectRinput_layer/input_layer/cms_group_id_embedding/cms_group_id_embedding_weights/TileXinput_layer/input_layer/cms_group_id_embedding/cms_group_id_embedding_weights/zeros_likeeinput_layer/input_layer/cms_group_id_embedding/cms_group_id_embedding_weights/embedding_lookup_sparse*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ü
Tinput_layer/input_layer/cms_group_id_embedding/cms_group_id_embedding_weights/Cast_1CastJinput_layer/input_layer/cms_group_id_embedding/to_sparse_input/dense_shape*

SrcT0	*
_output_shapes
:*

DstT0
Ľ
[input_layer/input_layer/cms_group_id_embedding/cms_group_id_embedding_weights/Slice_1/beginConst*
dtype0*
valueB: *
_output_shapes
:
¤
Zinput_layer/input_layer/cms_group_id_embedding/cms_group_id_embedding_weights/Slice_1/sizeConst*
valueB:*
_output_shapes
:*
dtype0

Uinput_layer/input_layer/cms_group_id_embedding/cms_group_id_embedding_weights/Slice_1SliceTinput_layer/input_layer/cms_group_id_embedding/cms_group_id_embedding_weights/Cast_1[input_layer/input_layer/cms_group_id_embedding/cms_group_id_embedding_weights/Slice_1/beginZinput_layer/input_layer/cms_group_id_embedding/cms_group_id_embedding_weights/Slice_1/size*
T0*
_output_shapes
:*
Index0
Ň
Uinput_layer/input_layer/cms_group_id_embedding/cms_group_id_embedding_weights/Shape_1ShapeMinput_layer/input_layer/cms_group_id_embedding/cms_group_id_embedding_weights*
T0*
_output_shapes
:
Ľ
[input_layer/input_layer/cms_group_id_embedding/cms_group_id_embedding_weights/Slice_2/beginConst*
dtype0*
_output_shapes
:*
valueB:
­
Zinput_layer/input_layer/cms_group_id_embedding/cms_group_id_embedding_weights/Slice_2/sizeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
_output_shapes
:*
dtype0
 
Uinput_layer/input_layer/cms_group_id_embedding/cms_group_id_embedding_weights/Slice_2SliceUinput_layer/input_layer/cms_group_id_embedding/cms_group_id_embedding_weights/Shape_1[input_layer/input_layer/cms_group_id_embedding/cms_group_id_embedding_weights/Slice_2/beginZinput_layer/input_layer/cms_group_id_embedding/cms_group_id_embedding_weights/Slice_2/size*
Index0*
T0*
_output_shapes
:

Yinput_layer/input_layer/cms_group_id_embedding/cms_group_id_embedding_weights/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 

Tinput_layer/input_layer/cms_group_id_embedding/cms_group_id_embedding_weights/concatConcatV2Uinput_layer/input_layer/cms_group_id_embedding/cms_group_id_embedding_weights/Slice_1Uinput_layer/input_layer/cms_group_id_embedding/cms_group_id_embedding_weights/Slice_2Yinput_layer/input_layer/cms_group_id_embedding/cms_group_id_embedding_weights/concat/axis*
T0*
N*
_output_shapes
:
š
Winput_layer/input_layer/cms_group_id_embedding/cms_group_id_embedding_weights/Reshape_2ReshapeMinput_layer/input_layer/cms_group_id_embedding/cms_group_id_embedding_weightsTinput_layer/input_layer/cms_group_id_embedding/cms_group_id_embedding_weights/concat*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
ť
4input_layer/input_layer/cms_group_id_embedding/ShapeShapeWinput_layer/input_layer/cms_group_id_embedding/cms_group_id_embedding_weights/Reshape_2*
T0*
_output_shapes
:

Binput_layer/input_layer/cms_group_id_embedding/strided_slice/stackConst*
valueB: *
_output_shapes
:*
dtype0

Dinput_layer/input_layer/cms_group_id_embedding/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:

Dinput_layer/input_layer/cms_group_id_embedding/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:

<input_layer/input_layer/cms_group_id_embedding/strided_sliceStridedSlice4input_layer/input_layer/cms_group_id_embedding/ShapeBinput_layer/input_layer/cms_group_id_embedding/strided_slice/stackDinput_layer/input_layer/cms_group_id_embedding/strided_slice/stack_1Dinput_layer/input_layer/cms_group_id_embedding/strided_slice/stack_2*
T0*
Index0*
_output_shapes
: *
shrink_axis_mask

>input_layer/input_layer/cms_group_id_embedding/Reshape/shape/1Const*
_output_shapes
: *
value	B :*
dtype0
đ
<input_layer/input_layer/cms_group_id_embedding/Reshape/shapePack<input_layer/input_layer/cms_group_id_embedding/strided_slice>input_layer/input_layer/cms_group_id_embedding/Reshape/shape/1*
N*
T0*
_output_shapes
:

6input_layer/input_layer/cms_group_id_embedding/ReshapeReshapeWinput_layer/input_layer/cms_group_id_embedding/cms_group_id_embedding_weights/Reshape_2<input_layer/input_layer/cms_group_id_embedding/Reshape/shape*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

:input_layer/input_layer/cms_segid_embedding/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙
Á
6input_layer/input_layer/cms_segid_embedding/ExpandDims
ExpandDimsPlaceholder_9:input_layer/input_layer/cms_segid_embedding/ExpandDims/dim*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

Jinput_layer/input_layer/cms_segid_embedding/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
valueB B *
dtype0

Dinput_layer/input_layer/cms_segid_embedding/to_sparse_input/NotEqualNotEqual6input_layer/input_layer/cms_segid_embedding/ExpandDimsJinput_layer/input_layer/cms_segid_embedding/to_sparse_input/ignore_value/x*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
ť
Cinput_layer/input_layer/cms_segid_embedding/to_sparse_input/indicesWhereDinput_layer/input_layer/cms_segid_embedding/to_sparse_input/NotEqual*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

Binput_layer/input_layer/cms_segid_embedding/to_sparse_input/valuesGatherNd6input_layer/input_layer/cms_segid_embedding/ExpandDimsCinput_layer/input_layer/cms_segid_embedding/to_sparse_input/indices*
Tparams0*
Tindices0	*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
˝
Ginput_layer/input_layer/cms_segid_embedding/to_sparse_input/dense_shapeShape6input_layer/input_layer/cms_segid_embedding/ExpandDims*
out_type0	*
T0*
_output_shapes
:
Č
2input_layer/input_layer/cms_segid_embedding/lookupStringToHashBucketFastBinput_layer/input_layer/cms_segid_embedding/to_sparse_input/values*
num_bucketsd*#
_output_shapes
:˙˙˙˙˙˙˙˙˙

ginput_layer/input_layer/cms_segid_embedding/embedding_weights/part_0/Initializer/truncated_normal/shapeConst*
dtype0*
_output_shapes
:*
valueB"d      *W
_classM
KIloc:@input_layer/input_layer/cms_segid_embedding/embedding_weights/part_0

finput_layer/input_layer/cms_segid_embedding/embedding_weights/part_0/Initializer/truncated_normal/meanConst*
dtype0*
_output_shapes
: *W
_classM
KIloc:@input_layer/input_layer/cms_segid_embedding/embedding_weights/part_0*
valueB
 *    

hinput_layer/input_layer/cms_segid_embedding/embedding_weights/part_0/Initializer/truncated_normal/stddevConst*
dtype0*
_output_shapes
: *
valueB
 *  >*W
_classM
KIloc:@input_layer/input_layer/cms_segid_embedding/embedding_weights/part_0
ü
qinput_layer/input_layer/cms_segid_embedding/embedding_weights/part_0/Initializer/truncated_normal/TruncatedNormalTruncatedNormalginput_layer/input_layer/cms_segid_embedding/embedding_weights/part_0/Initializer/truncated_normal/shape*
_output_shapes

:d*
T0*W
_classM
KIloc:@input_layer/input_layer/cms_segid_embedding/embedding_weights/part_0*
dtype0
Ë
einput_layer/input_layer/cms_segid_embedding/embedding_weights/part_0/Initializer/truncated_normal/mulMulqinput_layer/input_layer/cms_segid_embedding/embedding_weights/part_0/Initializer/truncated_normal/TruncatedNormalhinput_layer/input_layer/cms_segid_embedding/embedding_weights/part_0/Initializer/truncated_normal/stddev*W
_classM
KIloc:@input_layer/input_layer/cms_segid_embedding/embedding_weights/part_0*
T0*
_output_shapes

:d
š
ainput_layer/input_layer/cms_segid_embedding/embedding_weights/part_0/Initializer/truncated_normalAddeinput_layer/input_layer/cms_segid_embedding/embedding_weights/part_0/Initializer/truncated_normal/mulfinput_layer/input_layer/cms_segid_embedding/embedding_weights/part_0/Initializer/truncated_normal/mean*
_output_shapes

:d*
T0*W
_classM
KIloc:@input_layer/input_layer/cms_segid_embedding/embedding_weights/part_0
í
Dinput_layer/input_layer/cms_segid_embedding/embedding_weights/part_0
VariableV2*
dtype0*
_output_shapes

:d*
shape
:d*W
_classM
KIloc:@input_layer/input_layer/cms_segid_embedding/embedding_weights/part_0

Kinput_layer/input_layer/cms_segid_embedding/embedding_weights/part_0/AssignAssignDinput_layer/input_layer/cms_segid_embedding/embedding_weights/part_0ainput_layer/input_layer/cms_segid_embedding/embedding_weights/part_0/Initializer/truncated_normal*
T0*W
_classM
KIloc:@input_layer/input_layer/cms_segid_embedding/embedding_weights/part_0*
_output_shapes

:d

Iinput_layer/input_layer/cms_segid_embedding/embedding_weights/part_0/readIdentityDinput_layer/input_layer/cms_segid_embedding/embedding_weights/part_0*
T0*W
_classM
KIloc:@input_layer/input_layer/cms_segid_embedding/embedding_weights/part_0*
_output_shapes

:d

Sinput_layer/input_layer/cms_segid_embedding/cms_segid_embedding_weights/Slice/beginConst*
dtype0*
valueB: *
_output_shapes
:

Rinput_layer/input_layer/cms_segid_embedding/cms_segid_embedding_weights/Slice/sizeConst*
dtype0*
_output_shapes
:*
valueB:
ú
Minput_layer/input_layer/cms_segid_embedding/cms_segid_embedding_weights/SliceSliceGinput_layer/input_layer/cms_segid_embedding/to_sparse_input/dense_shapeSinput_layer/input_layer/cms_segid_embedding/cms_segid_embedding_weights/Slice/beginRinput_layer/input_layer/cms_segid_embedding/cms_segid_embedding_weights/Slice/size*
Index0*
_output_shapes
:*
T0	

Minput_layer/input_layer/cms_segid_embedding/cms_segid_embedding_weights/ConstConst*
_output_shapes
:*
valueB: *
dtype0

Linput_layer/input_layer/cms_segid_embedding/cms_segid_embedding_weights/ProdProdMinput_layer/input_layer/cms_segid_embedding/cms_segid_embedding_weights/SliceMinput_layer/input_layer/cms_segid_embedding/cms_segid_embedding_weights/Const*
_output_shapes
: *
T0	

Xinput_layer/input_layer/cms_segid_embedding/cms_segid_embedding_weights/GatherV2/indicesConst*
value	B :*
_output_shapes
: *
dtype0

Uinput_layer/input_layer/cms_segid_embedding/cms_segid_embedding_weights/GatherV2/axisConst*
_output_shapes
: *
value	B : *
dtype0

Pinput_layer/input_layer/cms_segid_embedding/cms_segid_embedding_weights/GatherV2GatherV2Ginput_layer/input_layer/cms_segid_embedding/to_sparse_input/dense_shapeXinput_layer/input_layer/cms_segid_embedding/cms_segid_embedding_weights/GatherV2/indicesUinput_layer/input_layer/cms_segid_embedding/cms_segid_embedding_weights/GatherV2/axis*
Tparams0	*
Tindices0*
Taxis0*
_output_shapes
: 
¤
Ninput_layer/input_layer/cms_segid_embedding/cms_segid_embedding_weights/Cast/xPackLinput_layer/input_layer/cms_segid_embedding/cms_segid_embedding_weights/ProdPinput_layer/input_layer/cms_segid_embedding/cms_segid_embedding_weights/GatherV2*
N*
T0	*
_output_shapes
:
ó
Uinput_layer/input_layer/cms_segid_embedding/cms_segid_embedding_weights/SparseReshapeSparseReshapeCinput_layer/input_layer/cms_segid_embedding/to_sparse_input/indicesGinput_layer/input_layer/cms_segid_embedding/to_sparse_input/dense_shapeNinput_layer/input_layer/cms_segid_embedding/cms_segid_embedding_weights/Cast/x*-
_output_shapes
:˙˙˙˙˙˙˙˙˙:
Ě
^input_layer/input_layer/cms_segid_embedding/cms_segid_embedding_weights/SparseReshape/IdentityIdentity2input_layer/input_layer/cms_segid_embedding/lookup*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0	

Vinput_layer/input_layer/cms_segid_embedding/cms_segid_embedding_weights/GreaterEqual/yConst*
dtype0	*
_output_shapes
: *
value	B	 R 
Ę
Tinput_layer/input_layer/cms_segid_embedding/cms_segid_embedding_weights/GreaterEqualGreaterEqual^input_layer/input_layer/cms_segid_embedding/cms_segid_embedding_weights/SparseReshape/IdentityVinput_layer/input_layer/cms_segid_embedding/cms_segid_embedding_weights/GreaterEqual/y*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0	
Ő
Minput_layer/input_layer/cms_segid_embedding/cms_segid_embedding_weights/WhereWhereTinput_layer/input_layer/cms_segid_embedding/cms_segid_embedding_weights/GreaterEqual*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
¨
Uinput_layer/input_layer/cms_segid_embedding/cms_segid_embedding_weights/Reshape/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
Ž
Oinput_layer/input_layer/cms_segid_embedding/cms_segid_embedding_weights/ReshapeReshapeMinput_layer/input_layer/cms_segid_embedding/cms_segid_embedding_weights/WhereUinput_layer/input_layer/cms_segid_embedding/cms_segid_embedding_weights/Reshape/shape*
T0	*#
_output_shapes
:˙˙˙˙˙˙˙˙˙

Winput_layer/input_layer/cms_segid_embedding/cms_segid_embedding_weights/GatherV2_1/axisConst*
dtype0*
value	B : *
_output_shapes
: 
´
Rinput_layer/input_layer/cms_segid_embedding/cms_segid_embedding_weights/GatherV2_1GatherV2Uinput_layer/input_layer/cms_segid_embedding/cms_segid_embedding_weights/SparseReshapeOinput_layer/input_layer/cms_segid_embedding/cms_segid_embedding_weights/ReshapeWinput_layer/input_layer/cms_segid_embedding/cms_segid_embedding_weights/GatherV2_1/axis*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
Tindices0	*
Taxis0*
Tparams0	

Winput_layer/input_layer/cms_segid_embedding/cms_segid_embedding_weights/GatherV2_2/axisConst*
dtype0*
value	B : *
_output_shapes
: 
š
Rinput_layer/input_layer/cms_segid_embedding/cms_segid_embedding_weights/GatherV2_2GatherV2^input_layer/input_layer/cms_segid_embedding/cms_segid_embedding_weights/SparseReshape/IdentityOinput_layer/input_layer/cms_segid_embedding/cms_segid_embedding_weights/ReshapeWinput_layer/input_layer/cms_segid_embedding/cms_segid_embedding_weights/GatherV2_2/axis*
Taxis0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
Tindices0	*
Tparams0	
Ú
Pinput_layer/input_layer/cms_segid_embedding/cms_segid_embedding_weights/IdentityIdentityWinput_layer/input_layer/cms_segid_embedding/cms_segid_embedding_weights/SparseReshape:1*
T0	*
_output_shapes
:
Ł
ainput_layer/input_layer/cms_segid_embedding/cms_segid_embedding_weights/SparseFillEmptyRows/ConstConst*
dtype0	*
value	B	 R *
_output_shapes
: 
Â
oinput_layer/input_layer/cms_segid_embedding/cms_segid_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowsSparseFillEmptyRowsRinput_layer/input_layer/cms_segid_embedding/cms_segid_embedding_weights/GatherV2_1Rinput_layer/input_layer/cms_segid_embedding/cms_segid_embedding_weights/GatherV2_2Pinput_layer/input_layer/cms_segid_embedding/cms_segid_embedding_weights/Identityainput_layer/input_layer/cms_segid_embedding/cms_segid_embedding_weights/SparseFillEmptyRows/Const*T
_output_shapesB
@:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0	
Ä
sinput_layer/input_layer/cms_segid_embedding/cms_segid_embedding_weights/embedding_lookup_sparse/strided_slice/stackConst*
valueB"        *
_output_shapes
:*
dtype0
Ć
uinput_layer/input_layer/cms_segid_embedding/cms_segid_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB"       
Ć
uinput_layer/input_layer/cms_segid_embedding/cms_segid_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
Ć
minput_layer/input_layer/cms_segid_embedding/cms_segid_embedding_weights/embedding_lookup_sparse/strided_sliceStridedSliceoinput_layer/input_layer/cms_segid_embedding/cms_segid_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowssinput_layer/input_layer/cms_segid_embedding/cms_segid_embedding_weights/embedding_lookup_sparse/strided_slice/stackuinput_layer/input_layer/cms_segid_embedding/cms_segid_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1uinput_layer/input_layer/cms_segid_embedding/cms_segid_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
Index0*
end_mask*

begin_mask*
T0	*
shrink_axis_mask

dinput_layer/input_layer/cms_segid_embedding/cms_segid_embedding_weights/embedding_lookup_sparse/CastCastminput_layer/input_layer/cms_segid_embedding/cms_segid_embedding_weights/embedding_lookup_sparse/strided_slice*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*

SrcT0	*

DstT0
 
finput_layer/input_layer/cms_segid_embedding/cms_segid_embedding_weights/embedding_lookup_sparse/UniqueUniqueqinput_layer/input_layer/cms_segid_embedding/cms_segid_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:1*
T0	*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙

uinput_layer/input_layer/cms_segid_embedding/cms_segid_embedding_weights/embedding_lookup_sparse/embedding_lookup/axisConst*W
_classM
KIloc:@input_layer/input_layer/cms_segid_embedding/embedding_weights/part_0*
value	B : *
_output_shapes
: *
dtype0
Ô
pinput_layer/input_layer/cms_segid_embedding/cms_segid_embedding_weights/embedding_lookup_sparse/embedding_lookupGatherV2Iinput_layer/input_layer/cms_segid_embedding/embedding_weights/part_0/readfinput_layer/input_layer/cms_segid_embedding/cms_segid_embedding_weights/embedding_lookup_sparse/Uniqueuinput_layer/input_layer/cms_segid_embedding/cms_segid_embedding_weights/embedding_lookup_sparse/embedding_lookup/axis*W
_classM
KIloc:@input_layer/input_layer/cms_segid_embedding/embedding_weights/part_0*
Tparams0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
Taxis0*
Tindices0	
Š
yinput_layer/input_layer/cms_segid_embedding/cms_segid_embedding_weights/embedding_lookup_sparse/embedding_lookup/IdentityIdentitypinput_layer/input_layer/cms_segid_embedding/cms_segid_embedding_weights/embedding_lookup_sparse/embedding_lookup*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
ń
_input_layer/input_layer/cms_segid_embedding/cms_segid_embedding_weights/embedding_lookup_sparseSparseSegmentMeanyinput_layer/input_layer/cms_segid_embedding/cms_segid_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identityhinput_layer/input_layer/cms_segid_embedding/cms_segid_embedding_weights/embedding_lookup_sparse/Unique:1dinput_layer/input_layer/cms_segid_embedding/cms_segid_embedding_weights/embedding_lookup_sparse/Cast*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
¨
Winput_layer/input_layer/cms_segid_embedding/cms_segid_embedding_weights/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   
Ú
Qinput_layer/input_layer/cms_segid_embedding/cms_segid_embedding_weights/Reshape_1Reshapeqinput_layer/input_layer/cms_segid_embedding/cms_segid_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:2Winput_layer/input_layer/cms_segid_embedding/cms_segid_embedding_weights/Reshape_1/shape*
T0
*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ü
Minput_layer/input_layer/cms_segid_embedding/cms_segid_embedding_weights/ShapeShape_input_layer/input_layer/cms_segid_embedding/cms_segid_embedding_weights/embedding_lookup_sparse*
_output_shapes
:*
T0
Ľ
[input_layer/input_layer/cms_segid_embedding/cms_segid_embedding_weights/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
§
]input_layer/input_layer/cms_segid_embedding/cms_segid_embedding_weights/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
§
]input_layer/input_layer/cms_segid_embedding/cms_segid_embedding_weights/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:

Uinput_layer/input_layer/cms_segid_embedding/cms_segid_embedding_weights/strided_sliceStridedSliceMinput_layer/input_layer/cms_segid_embedding/cms_segid_embedding_weights/Shape[input_layer/input_layer/cms_segid_embedding/cms_segid_embedding_weights/strided_slice/stack]input_layer/input_layer/cms_segid_embedding/cms_segid_embedding_weights/strided_slice/stack_1]input_layer/input_layer/cms_segid_embedding/cms_segid_embedding_weights/strided_slice/stack_2*
Index0*
_output_shapes
: *
T0*
shrink_axis_mask

Oinput_layer/input_layer/cms_segid_embedding/cms_segid_embedding_weights/stack/0Const*
dtype0*
value	B :*
_output_shapes
: 
Ť
Minput_layer/input_layer/cms_segid_embedding/cms_segid_embedding_weights/stackPackOinput_layer/input_layer/cms_segid_embedding/cms_segid_embedding_weights/stack/0Uinput_layer/input_layer/cms_segid_embedding/cms_segid_embedding_weights/strided_slice*
_output_shapes
:*
N*
T0
ą
Linput_layer/input_layer/cms_segid_embedding/cms_segid_embedding_weights/TileTileQinput_layer/input_layer/cms_segid_embedding/cms_segid_embedding_weights/Reshape_1Minput_layer/input_layer/cms_segid_embedding/cms_segid_embedding_weights/stack*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
T0

ň
Rinput_layer/input_layer/cms_segid_embedding/cms_segid_embedding_weights/zeros_like	ZerosLike_input_layer/input_layer/cms_segid_embedding/cms_segid_embedding_weights/embedding_lookup_sparse*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

Ginput_layer/input_layer/cms_segid_embedding/cms_segid_embedding_weightsSelectLinput_layer/input_layer/cms_segid_embedding/cms_segid_embedding_weights/TileRinput_layer/input_layer/cms_segid_embedding/cms_segid_embedding_weights/zeros_like_input_layer/input_layer/cms_segid_embedding/cms_segid_embedding_weights/embedding_lookup_sparse*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ó
Ninput_layer/input_layer/cms_segid_embedding/cms_segid_embedding_weights/Cast_1CastGinput_layer/input_layer/cms_segid_embedding/to_sparse_input/dense_shape*

SrcT0	*

DstT0*
_output_shapes
:

Uinput_layer/input_layer/cms_segid_embedding/cms_segid_embedding_weights/Slice_1/beginConst*
valueB: *
_output_shapes
:*
dtype0

Tinput_layer/input_layer/cms_segid_embedding/cms_segid_embedding_weights/Slice_1/sizeConst*
_output_shapes
:*
valueB:*
dtype0

Oinput_layer/input_layer/cms_segid_embedding/cms_segid_embedding_weights/Slice_1SliceNinput_layer/input_layer/cms_segid_embedding/cms_segid_embedding_weights/Cast_1Uinput_layer/input_layer/cms_segid_embedding/cms_segid_embedding_weights/Slice_1/beginTinput_layer/input_layer/cms_segid_embedding/cms_segid_embedding_weights/Slice_1/size*
Index0*
_output_shapes
:*
T0
Ć
Oinput_layer/input_layer/cms_segid_embedding/cms_segid_embedding_weights/Shape_1ShapeGinput_layer/input_layer/cms_segid_embedding/cms_segid_embedding_weights*
_output_shapes
:*
T0

Uinput_layer/input_layer/cms_segid_embedding/cms_segid_embedding_weights/Slice_2/beginConst*
valueB:*
dtype0*
_output_shapes
:
§
Tinput_layer/input_layer/cms_segid_embedding/cms_segid_embedding_weights/Slice_2/sizeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:

Oinput_layer/input_layer/cms_segid_embedding/cms_segid_embedding_weights/Slice_2SliceOinput_layer/input_layer/cms_segid_embedding/cms_segid_embedding_weights/Shape_1Uinput_layer/input_layer/cms_segid_embedding/cms_segid_embedding_weights/Slice_2/beginTinput_layer/input_layer/cms_segid_embedding/cms_segid_embedding_weights/Slice_2/size*
Index0*
T0*
_output_shapes
:

Sinput_layer/input_layer/cms_segid_embedding/cms_segid_embedding_weights/concat/axisConst*
_output_shapes
: *
value	B : *
dtype0
˙
Ninput_layer/input_layer/cms_segid_embedding/cms_segid_embedding_weights/concatConcatV2Oinput_layer/input_layer/cms_segid_embedding/cms_segid_embedding_weights/Slice_1Oinput_layer/input_layer/cms_segid_embedding/cms_segid_embedding_weights/Slice_2Sinput_layer/input_layer/cms_segid_embedding/cms_segid_embedding_weights/concat/axis*
_output_shapes
:*
N*
T0
§
Qinput_layer/input_layer/cms_segid_embedding/cms_segid_embedding_weights/Reshape_2ReshapeGinput_layer/input_layer/cms_segid_embedding/cms_segid_embedding_weightsNinput_layer/input_layer/cms_segid_embedding/cms_segid_embedding_weights/concat*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
˛
1input_layer/input_layer/cms_segid_embedding/ShapeShapeQinput_layer/input_layer/cms_segid_embedding/cms_segid_embedding_weights/Reshape_2*
T0*
_output_shapes
:

?input_layer/input_layer/cms_segid_embedding/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 

Ainput_layer/input_layer/cms_segid_embedding/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:

Ainput_layer/input_layer/cms_segid_embedding/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:

9input_layer/input_layer/cms_segid_embedding/strided_sliceStridedSlice1input_layer/input_layer/cms_segid_embedding/Shape?input_layer/input_layer/cms_segid_embedding/strided_slice/stackAinput_layer/input_layer/cms_segid_embedding/strided_slice/stack_1Ainput_layer/input_layer/cms_segid_embedding/strided_slice/stack_2*
T0*
shrink_axis_mask*
Index0*
_output_shapes
: 
}
;input_layer/input_layer/cms_segid_embedding/Reshape/shape/1Const*
_output_shapes
: *
value	B :*
dtype0
ç
9input_layer/input_layer/cms_segid_embedding/Reshape/shapePack9input_layer/input_layer/cms_segid_embedding/strided_slice;input_layer/input_layer/cms_segid_embedding/Reshape/shape/1*
_output_shapes
:*
N*
T0
ţ
3input_layer/input_layer/cms_segid_embedding/ReshapeReshapeQinput_layer/input_layer/cms_segid_embedding/cms_segid_embedding_weights/Reshape_29input_layer/input_layer/cms_segid_embedding/Reshape/shape*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

Einput_layer/input_layer/new_user_class_level_embedding/ExpandDims/dimConst*
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙*
_output_shapes
: 
Ř
Ainput_layer/input_layer/new_user_class_level_embedding/ExpandDims
ExpandDimsPlaceholder_16Einput_layer/input_layer/new_user_class_level_embedding/ExpandDims/dim*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

Uinput_layer/input_layer/new_user_class_level_embedding/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB B 
§
Oinput_layer/input_layer/new_user_class_level_embedding/to_sparse_input/NotEqualNotEqualAinput_layer/input_layer/new_user_class_level_embedding/ExpandDimsUinput_layer/input_layer/new_user_class_level_embedding/to_sparse_input/ignore_value/x*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ń
Ninput_layer/input_layer/new_user_class_level_embedding/to_sparse_input/indicesWhereOinput_layer/input_layer/new_user_class_level_embedding/to_sparse_input/NotEqual*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
°
Minput_layer/input_layer/new_user_class_level_embedding/to_sparse_input/valuesGatherNdAinput_layer/input_layer/new_user_class_level_embedding/ExpandDimsNinput_layer/input_layer/new_user_class_level_embedding/to_sparse_input/indices*
Tparams0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
Tindices0	
Ó
Rinput_layer/input_layer/new_user_class_level_embedding/to_sparse_input/dense_shapeShapeAinput_layer/input_layer/new_user_class_level_embedding/ExpandDims*
T0*
out_type0	*
_output_shapes
:
Ţ
=input_layer/input_layer/new_user_class_level_embedding/lookupStringToHashBucketFastMinput_layer/input_layer/new_user_class_level_embedding/to_sparse_input/values*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
num_buckets

§
rinput_layer/input_layer/new_user_class_level_embedding/embedding_weights/part_0/Initializer/truncated_normal/shapeConst*
dtype0*
_output_shapes
:*
valueB"
      *b
_classX
VTloc:@input_layer/input_layer/new_user_class_level_embedding/embedding_weights/part_0

qinput_layer/input_layer/new_user_class_level_embedding/embedding_weights/part_0/Initializer/truncated_normal/meanConst*b
_classX
VTloc:@input_layer/input_layer/new_user_class_level_embedding/embedding_weights/part_0*
valueB
 *    *
dtype0*
_output_shapes
: 

sinput_layer/input_layer/new_user_class_level_embedding/embedding_weights/part_0/Initializer/truncated_normal/stddevConst*
valueB
 *  >*
_output_shapes
: *b
_classX
VTloc:@input_layer/input_layer/new_user_class_level_embedding/embedding_weights/part_0*
dtype0

|input_layer/input_layer/new_user_class_level_embedding/embedding_weights/part_0/Initializer/truncated_normal/TruncatedNormalTruncatedNormalrinput_layer/input_layer/new_user_class_level_embedding/embedding_weights/part_0/Initializer/truncated_normal/shape*b
_classX
VTloc:@input_layer/input_layer/new_user_class_level_embedding/embedding_weights/part_0*
_output_shapes

:
*
dtype0*
T0
÷
pinput_layer/input_layer/new_user_class_level_embedding/embedding_weights/part_0/Initializer/truncated_normal/mulMul|input_layer/input_layer/new_user_class_level_embedding/embedding_weights/part_0/Initializer/truncated_normal/TruncatedNormalsinput_layer/input_layer/new_user_class_level_embedding/embedding_weights/part_0/Initializer/truncated_normal/stddev*
T0*
_output_shapes

:
*b
_classX
VTloc:@input_layer/input_layer/new_user_class_level_embedding/embedding_weights/part_0
ĺ
linput_layer/input_layer/new_user_class_level_embedding/embedding_weights/part_0/Initializer/truncated_normalAddpinput_layer/input_layer/new_user_class_level_embedding/embedding_weights/part_0/Initializer/truncated_normal/mulqinput_layer/input_layer/new_user_class_level_embedding/embedding_weights/part_0/Initializer/truncated_normal/mean*
_output_shapes

:
*
T0*b
_classX
VTloc:@input_layer/input_layer/new_user_class_level_embedding/embedding_weights/part_0

Oinput_layer/input_layer/new_user_class_level_embedding/embedding_weights/part_0
VariableV2*
dtype0*b
_classX
VTloc:@input_layer/input_layer/new_user_class_level_embedding/embedding_weights/part_0*
shape
:
*
_output_shapes

:

Ź
Vinput_layer/input_layer/new_user_class_level_embedding/embedding_weights/part_0/AssignAssignOinput_layer/input_layer/new_user_class_level_embedding/embedding_weights/part_0linput_layer/input_layer/new_user_class_level_embedding/embedding_weights/part_0/Initializer/truncated_normal*
_output_shapes

:
*b
_classX
VTloc:@input_layer/input_layer/new_user_class_level_embedding/embedding_weights/part_0*
T0
ž
Tinput_layer/input_layer/new_user_class_level_embedding/embedding_weights/part_0/readIdentityOinput_layer/input_layer/new_user_class_level_embedding/embedding_weights/part_0*
_output_shapes

:
*b
_classX
VTloc:@input_layer/input_layer/new_user_class_level_embedding/embedding_weights/part_0*
T0
ł
iinput_layer/input_layer/new_user_class_level_embedding/new_user_class_level_embedding_weights/Slice/beginConst*
dtype0*
_output_shapes
:*
valueB: 
˛
hinput_layer/input_layer/new_user_class_level_embedding/new_user_class_level_embedding_weights/Slice/sizeConst*
_output_shapes
:*
dtype0*
valueB:
Ç
cinput_layer/input_layer/new_user_class_level_embedding/new_user_class_level_embedding_weights/SliceSliceRinput_layer/input_layer/new_user_class_level_embedding/to_sparse_input/dense_shapeiinput_layer/input_layer/new_user_class_level_embedding/new_user_class_level_embedding_weights/Slice/beginhinput_layer/input_layer/new_user_class_level_embedding/new_user_class_level_embedding_weights/Slice/size*
_output_shapes
:*
Index0*
T0	
­
cinput_layer/input_layer/new_user_class_level_embedding/new_user_class_level_embedding_weights/ConstConst*
valueB: *
dtype0*
_output_shapes
:
Ő
binput_layer/input_layer/new_user_class_level_embedding/new_user_class_level_embedding_weights/ProdProdcinput_layer/input_layer/new_user_class_level_embedding/new_user_class_level_embedding_weights/Slicecinput_layer/input_layer/new_user_class_level_embedding/new_user_class_level_embedding_weights/Const*
_output_shapes
: *
T0	
°
ninput_layer/input_layer/new_user_class_level_embedding/new_user_class_level_embedding_weights/GatherV2/indicesConst*
_output_shapes
: *
value	B :*
dtype0
­
kinput_layer/input_layer/new_user_class_level_embedding/new_user_class_level_embedding_weights/GatherV2/axisConst*
value	B : *
_output_shapes
: *
dtype0
ç
finput_layer/input_layer/new_user_class_level_embedding/new_user_class_level_embedding_weights/GatherV2GatherV2Rinput_layer/input_layer/new_user_class_level_embedding/to_sparse_input/dense_shapeninput_layer/input_layer/new_user_class_level_embedding/new_user_class_level_embedding_weights/GatherV2/indiceskinput_layer/input_layer/new_user_class_level_embedding/new_user_class_level_embedding_weights/GatherV2/axis*
_output_shapes
: *
Tparams0	*
Taxis0*
Tindices0
ć
dinput_layer/input_layer/new_user_class_level_embedding/new_user_class_level_embedding_weights/Cast/xPackbinput_layer/input_layer/new_user_class_level_embedding/new_user_class_level_embedding_weights/Prodfinput_layer/input_layer/new_user_class_level_embedding/new_user_class_level_embedding_weights/GatherV2*
_output_shapes
:*
N*
T0	
ľ
kinput_layer/input_layer/new_user_class_level_embedding/new_user_class_level_embedding_weights/SparseReshapeSparseReshapeNinput_layer/input_layer/new_user_class_level_embedding/to_sparse_input/indicesRinput_layer/input_layer/new_user_class_level_embedding/to_sparse_input/dense_shapedinput_layer/input_layer/new_user_class_level_embedding/new_user_class_level_embedding_weights/Cast/x*-
_output_shapes
:˙˙˙˙˙˙˙˙˙:
í
tinput_layer/input_layer/new_user_class_level_embedding/new_user_class_level_embedding_weights/SparseReshape/IdentityIdentity=input_layer/input_layer/new_user_class_level_embedding/lookup*
T0	*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ž
linput_layer/input_layer/new_user_class_level_embedding/new_user_class_level_embedding_weights/GreaterEqual/yConst*
dtype0	*
value	B	 R *
_output_shapes
: 

jinput_layer/input_layer/new_user_class_level_embedding/new_user_class_level_embedding_weights/GreaterEqualGreaterEqualtinput_layer/input_layer/new_user_class_level_embedding/new_user_class_level_embedding_weights/SparseReshape/Identitylinput_layer/input_layer/new_user_class_level_embedding/new_user_class_level_embedding_weights/GreaterEqual/y*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0	

cinput_layer/input_layer/new_user_class_level_embedding/new_user_class_level_embedding_weights/WhereWherejinput_layer/input_layer/new_user_class_level_embedding/new_user_class_level_embedding_weights/GreaterEqual*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
ž
kinput_layer/input_layer/new_user_class_level_embedding/new_user_class_level_embedding_weights/Reshape/shapeConst*
dtype0*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙
đ
einput_layer/input_layer/new_user_class_level_embedding/new_user_class_level_embedding_weights/ReshapeReshapecinput_layer/input_layer/new_user_class_level_embedding/new_user_class_level_embedding_weights/Wherekinput_layer/input_layer/new_user_class_level_embedding/new_user_class_level_embedding_weights/Reshape/shape*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0	
Ż
minput_layer/input_layer/new_user_class_level_embedding/new_user_class_level_embedding_weights/GatherV2_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 

hinput_layer/input_layer/new_user_class_level_embedding/new_user_class_level_embedding_weights/GatherV2_1GatherV2kinput_layer/input_layer/new_user_class_level_embedding/new_user_class_level_embedding_weights/SparseReshapeeinput_layer/input_layer/new_user_class_level_embedding/new_user_class_level_embedding_weights/Reshapeminput_layer/input_layer/new_user_class_level_embedding/new_user_class_level_embedding_weights/GatherV2_1/axis*
Taxis0*
Tparams0	*
Tindices0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ż
minput_layer/input_layer/new_user_class_level_embedding/new_user_class_level_embedding_weights/GatherV2_2/axisConst*
dtype0*
value	B : *
_output_shapes
: 

hinput_layer/input_layer/new_user_class_level_embedding/new_user_class_level_embedding_weights/GatherV2_2GatherV2tinput_layer/input_layer/new_user_class_level_embedding/new_user_class_level_embedding_weights/SparseReshape/Identityeinput_layer/input_layer/new_user_class_level_embedding/new_user_class_level_embedding_weights/Reshapeminput_layer/input_layer/new_user_class_level_embedding/new_user_class_level_embedding_weights/GatherV2_2/axis*
Tindices0	*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
Taxis0*
Tparams0	

finput_layer/input_layer/new_user_class_level_embedding/new_user_class_level_embedding_weights/IdentityIdentityminput_layer/input_layer/new_user_class_level_embedding/new_user_class_level_embedding_weights/SparseReshape:1*
T0	*
_output_shapes
:
š
winput_layer/input_layer/new_user_class_level_embedding/new_user_class_level_embedding_weights/SparseFillEmptyRows/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R 
ą
input_layer/input_layer/new_user_class_level_embedding/new_user_class_level_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowsSparseFillEmptyRowshinput_layer/input_layer/new_user_class_level_embedding/new_user_class_level_embedding_weights/GatherV2_1hinput_layer/input_layer/new_user_class_level_embedding/new_user_class_level_embedding_weights/GatherV2_2finput_layer/input_layer/new_user_class_level_embedding/new_user_class_level_embedding_weights/Identitywinput_layer/input_layer/new_user_class_level_embedding/new_user_class_level_embedding_weights/SparseFillEmptyRows/Const*T
_output_shapesB
@:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0	
Ű
input_layer/input_layer/new_user_class_level_embedding/new_user_class_level_embedding_weights/embedding_lookup_sparse/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        
Ý
input_layer/input_layer/new_user_class_level_embedding/new_user_class_level_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB"       
Ý
input_layer/input_layer/new_user_class_level_embedding/new_user_class_level_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2Const*
dtype0*
valueB"      *
_output_shapes
:
š
input_layer/input_layer/new_user_class_level_embedding/new_user_class_level_embedding_weights/embedding_lookup_sparse/strided_sliceStridedSliceinput_layer/input_layer/new_user_class_level_embedding/new_user_class_level_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowsinput_layer/input_layer/new_user_class_level_embedding/new_user_class_level_embedding_weights/embedding_lookup_sparse/strided_slice/stackinput_layer/input_layer/new_user_class_level_embedding/new_user_class_level_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1input_layer/input_layer/new_user_class_level_embedding/new_user_class_level_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2*
shrink_axis_mask*
end_mask*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
Index0*

begin_mask*
T0	
Ĺ
zinput_layer/input_layer/new_user_class_level_embedding/new_user_class_level_embedding_weights/embedding_lookup_sparse/CastCastinput_layer/input_layer/new_user_class_level_embedding/new_user_class_level_embedding_weights/embedding_lookup_sparse/strided_slice*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*

DstT0*

SrcT0	
Í
|input_layer/input_layer/new_user_class_level_embedding/new_user_class_level_embedding_weights/embedding_lookup_sparse/UniqueUniqueinput_layer/input_layer/new_user_class_level_embedding/new_user_class_level_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:1*
T0	*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
˛
input_layer/input_layer/new_user_class_level_embedding/new_user_class_level_embedding_weights/embedding_lookup_sparse/embedding_lookup/axisConst*b
_classX
VTloc:@input_layer/input_layer/new_user_class_level_embedding/embedding_weights/part_0*
value	B : *
dtype0*
_output_shapes
: 
Ž
input_layer/input_layer/new_user_class_level_embedding/new_user_class_level_embedding_weights/embedding_lookup_sparse/embedding_lookupGatherV2Tinput_layer/input_layer/new_user_class_level_embedding/embedding_weights/part_0/read|input_layer/input_layer/new_user_class_level_embedding/new_user_class_level_embedding_weights/embedding_lookup_sparse/Uniqueinput_layer/input_layer/new_user_class_level_embedding/new_user_class_level_embedding_weights/embedding_lookup_sparse/embedding_lookup/axis*
Tindices0	*
Taxis0*
Tparams0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*b
_classX
VTloc:@input_layer/input_layer/new_user_class_level_embedding/embedding_weights/part_0
×
input_layer/input_layer/new_user_class_level_embedding/new_user_class_level_embedding_weights/embedding_lookup_sparse/embedding_lookup/IdentityIdentityinput_layer/input_layer/new_user_class_level_embedding/new_user_class_level_embedding_weights/embedding_lookup_sparse/embedding_lookup*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ę
uinput_layer/input_layer/new_user_class_level_embedding/new_user_class_level_embedding_weights/embedding_lookup_sparseSparseSegmentMeaninput_layer/input_layer/new_user_class_level_embedding/new_user_class_level_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identity~input_layer/input_layer/new_user_class_level_embedding/new_user_class_level_embedding_weights/embedding_lookup_sparse/Unique:1zinput_layer/input_layer/new_user_class_level_embedding/new_user_class_level_embedding_weights/embedding_lookup_sparse/Cast*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
ž
minput_layer/input_layer/new_user_class_level_embedding/new_user_class_level_embedding_weights/Reshape_1/shapeConst*
_output_shapes
:*
valueB"˙˙˙˙   *
dtype0

ginput_layer/input_layer/new_user_class_level_embedding/new_user_class_level_embedding_weights/Reshape_1Reshapeinput_layer/input_layer/new_user_class_level_embedding/new_user_class_level_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:2minput_layer/input_layer/new_user_class_level_embedding/new_user_class_level_embedding_weights/Reshape_1/shape*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0


cinput_layer/input_layer/new_user_class_level_embedding/new_user_class_level_embedding_weights/ShapeShapeuinput_layer/input_layer/new_user_class_level_embedding/new_user_class_level_embedding_weights/embedding_lookup_sparse*
_output_shapes
:*
T0
ť
qinput_layer/input_layer/new_user_class_level_embedding/new_user_class_level_embedding_weights/strided_slice/stackConst*
dtype0*
valueB:*
_output_shapes
:
˝
sinput_layer/input_layer/new_user_class_level_embedding/new_user_class_level_embedding_weights/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
˝
sinput_layer/input_layer/new_user_class_level_embedding/new_user_class_level_embedding_weights/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:

kinput_layer/input_layer/new_user_class_level_embedding/new_user_class_level_embedding_weights/strided_sliceStridedSlicecinput_layer/input_layer/new_user_class_level_embedding/new_user_class_level_embedding_weights/Shapeqinput_layer/input_layer/new_user_class_level_embedding/new_user_class_level_embedding_weights/strided_slice/stacksinput_layer/input_layer/new_user_class_level_embedding/new_user_class_level_embedding_weights/strided_slice/stack_1sinput_layer/input_layer/new_user_class_level_embedding/new_user_class_level_embedding_weights/strided_slice/stack_2*
shrink_axis_mask*
_output_shapes
: *
T0*
Index0
§
einput_layer/input_layer/new_user_class_level_embedding/new_user_class_level_embedding_weights/stack/0Const*
_output_shapes
: *
value	B :*
dtype0
í
cinput_layer/input_layer/new_user_class_level_embedding/new_user_class_level_embedding_weights/stackPackeinput_layer/input_layer/new_user_class_level_embedding/new_user_class_level_embedding_weights/stack/0kinput_layer/input_layer/new_user_class_level_embedding/new_user_class_level_embedding_weights/strided_slice*
_output_shapes
:*
N*
T0
ó
binput_layer/input_layer/new_user_class_level_embedding/new_user_class_level_embedding_weights/TileTileginput_layer/input_layer/new_user_class_level_embedding/new_user_class_level_embedding_weights/Reshape_1cinput_layer/input_layer/new_user_class_level_embedding/new_user_class_level_embedding_weights/stack*
T0
*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙

hinput_layer/input_layer/new_user_class_level_embedding/new_user_class_level_embedding_weights/zeros_like	ZerosLikeuinput_layer/input_layer/new_user_class_level_embedding/new_user_class_level_embedding_weights/embedding_lookup_sparse*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ţ
]input_layer/input_layer/new_user_class_level_embedding/new_user_class_level_embedding_weightsSelectbinput_layer/input_layer/new_user_class_level_embedding/new_user_class_level_embedding_weights/Tilehinput_layer/input_layer/new_user_class_level_embedding/new_user_class_level_embedding_weights/zeros_likeuinput_layer/input_layer/new_user_class_level_embedding/new_user_class_level_embedding_weights/embedding_lookup_sparse*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
ô
dinput_layer/input_layer/new_user_class_level_embedding/new_user_class_level_embedding_weights/Cast_1CastRinput_layer/input_layer/new_user_class_level_embedding/to_sparse_input/dense_shape*

DstT0*
_output_shapes
:*

SrcT0	
ľ
kinput_layer/input_layer/new_user_class_level_embedding/new_user_class_level_embedding_weights/Slice_1/beginConst*
valueB: *
dtype0*
_output_shapes
:
´
jinput_layer/input_layer/new_user_class_level_embedding/new_user_class_level_embedding_weights/Slice_1/sizeConst*
dtype0*
valueB:*
_output_shapes
:
ß
einput_layer/input_layer/new_user_class_level_embedding/new_user_class_level_embedding_weights/Slice_1Slicedinput_layer/input_layer/new_user_class_level_embedding/new_user_class_level_embedding_weights/Cast_1kinput_layer/input_layer/new_user_class_level_embedding/new_user_class_level_embedding_weights/Slice_1/beginjinput_layer/input_layer/new_user_class_level_embedding/new_user_class_level_embedding_weights/Slice_1/size*
T0*
Index0*
_output_shapes
:
ň
einput_layer/input_layer/new_user_class_level_embedding/new_user_class_level_embedding_weights/Shape_1Shape]input_layer/input_layer/new_user_class_level_embedding/new_user_class_level_embedding_weights*
_output_shapes
:*
T0
ľ
kinput_layer/input_layer/new_user_class_level_embedding/new_user_class_level_embedding_weights/Slice_2/beginConst*
_output_shapes
:*
dtype0*
valueB:
˝
jinput_layer/input_layer/new_user_class_level_embedding/new_user_class_level_embedding_weights/Slice_2/sizeConst*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0
ŕ
einput_layer/input_layer/new_user_class_level_embedding/new_user_class_level_embedding_weights/Slice_2Sliceeinput_layer/input_layer/new_user_class_level_embedding/new_user_class_level_embedding_weights/Shape_1kinput_layer/input_layer/new_user_class_level_embedding/new_user_class_level_embedding_weights/Slice_2/beginjinput_layer/input_layer/new_user_class_level_embedding/new_user_class_level_embedding_weights/Slice_2/size*
_output_shapes
:*
T0*
Index0
Ť
iinput_layer/input_layer/new_user_class_level_embedding/new_user_class_level_embedding_weights/concat/axisConst*
dtype0*
_output_shapes
: *
value	B : 
×
dinput_layer/input_layer/new_user_class_level_embedding/new_user_class_level_embedding_weights/concatConcatV2einput_layer/input_layer/new_user_class_level_embedding/new_user_class_level_embedding_weights/Slice_1einput_layer/input_layer/new_user_class_level_embedding/new_user_class_level_embedding_weights/Slice_2iinput_layer/input_layer/new_user_class_level_embedding/new_user_class_level_embedding_weights/concat/axis*
T0*
_output_shapes
:*
N
é
ginput_layer/input_layer/new_user_class_level_embedding/new_user_class_level_embedding_weights/Reshape_2Reshape]input_layer/input_layer/new_user_class_level_embedding/new_user_class_level_embedding_weightsdinput_layer/input_layer/new_user_class_level_embedding/new_user_class_level_embedding_weights/concat*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ó
<input_layer/input_layer/new_user_class_level_embedding/ShapeShapeginput_layer/input_layer/new_user_class_level_embedding/new_user_class_level_embedding_weights/Reshape_2*
T0*
_output_shapes
:

Jinput_layer/input_layer/new_user_class_level_embedding/strided_slice/stackConst*
_output_shapes
:*
valueB: *
dtype0

Linput_layer/input_layer/new_user_class_level_embedding/strided_slice/stack_1Const*
valueB:*
_output_shapes
:*
dtype0

Linput_layer/input_layer/new_user_class_level_embedding/strided_slice/stack_2Const*
valueB:*
_output_shapes
:*
dtype0
Ŕ
Dinput_layer/input_layer/new_user_class_level_embedding/strided_sliceStridedSlice<input_layer/input_layer/new_user_class_level_embedding/ShapeJinput_layer/input_layer/new_user_class_level_embedding/strided_slice/stackLinput_layer/input_layer/new_user_class_level_embedding/strided_slice/stack_1Linput_layer/input_layer/new_user_class_level_embedding/strided_slice/stack_2*
_output_shapes
: *
shrink_axis_mask*
Index0*
T0

Finput_layer/input_layer/new_user_class_level_embedding/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :

Dinput_layer/input_layer/new_user_class_level_embedding/Reshape/shapePackDinput_layer/input_layer/new_user_class_level_embedding/strided_sliceFinput_layer/input_layer/new_user_class_level_embedding/Reshape/shape/1*
T0*
N*
_output_shapes
:
Ş
>input_layer/input_layer/new_user_class_level_embedding/ReshapeReshapeginput_layer/input_layer/new_user_class_level_embedding/new_user_class_level_embedding_weights/Reshape_2Dinput_layer/input_layer/new_user_class_level_embedding/Reshape/shape*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

;input_layer/input_layer/occupation_embedding/ExpandDims/dimConst*
dtype0*
_output_shapes
: *
valueB :
˙˙˙˙˙˙˙˙˙
Ä
7input_layer/input_layer/occupation_embedding/ExpandDims
ExpandDimsPlaceholder_15;input_layer/input_layer/occupation_embedding/ExpandDims/dim*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

Kinput_layer/input_layer/occupation_embedding/to_sparse_input/ignore_value/xConst*
dtype0*
valueB B *
_output_shapes
: 

Einput_layer/input_layer/occupation_embedding/to_sparse_input/NotEqualNotEqual7input_layer/input_layer/occupation_embedding/ExpandDimsKinput_layer/input_layer/occupation_embedding/to_sparse_input/ignore_value/x*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
˝
Dinput_layer/input_layer/occupation_embedding/to_sparse_input/indicesWhereEinput_layer/input_layer/occupation_embedding/to_sparse_input/NotEqual*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

Cinput_layer/input_layer/occupation_embedding/to_sparse_input/valuesGatherNd7input_layer/input_layer/occupation_embedding/ExpandDimsDinput_layer/input_layer/occupation_embedding/to_sparse_input/indices*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
Tparams0*
Tindices0	
ż
Hinput_layer/input_layer/occupation_embedding/to_sparse_input/dense_shapeShape7input_layer/input_layer/occupation_embedding/ExpandDims*
out_type0	*
_output_shapes
:*
T0
Ę
3input_layer/input_layer/occupation_embedding/lookupStringToHashBucketFastCinput_layer/input_layer/occupation_embedding/to_sparse_input/values*
num_buckets
*#
_output_shapes
:˙˙˙˙˙˙˙˙˙

hinput_layer/input_layer/occupation_embedding/embedding_weights/part_0/Initializer/truncated_normal/shapeConst*
valueB"
      *
_output_shapes
:*
dtype0*X
_classN
LJloc:@input_layer/input_layer/occupation_embedding/embedding_weights/part_0

ginput_layer/input_layer/occupation_embedding/embedding_weights/part_0/Initializer/truncated_normal/meanConst*
dtype0*
valueB
 *    *X
_classN
LJloc:@input_layer/input_layer/occupation_embedding/embedding_weights/part_0*
_output_shapes
: 

iinput_layer/input_layer/occupation_embedding/embedding_weights/part_0/Initializer/truncated_normal/stddevConst*X
_classN
LJloc:@input_layer/input_layer/occupation_embedding/embedding_weights/part_0*
_output_shapes
: *
valueB
 *  >*
dtype0
˙
rinput_layer/input_layer/occupation_embedding/embedding_weights/part_0/Initializer/truncated_normal/TruncatedNormalTruncatedNormalhinput_layer/input_layer/occupation_embedding/embedding_weights/part_0/Initializer/truncated_normal/shape*X
_classN
LJloc:@input_layer/input_layer/occupation_embedding/embedding_weights/part_0*
T0*
dtype0*
_output_shapes

:

Ď
finput_layer/input_layer/occupation_embedding/embedding_weights/part_0/Initializer/truncated_normal/mulMulrinput_layer/input_layer/occupation_embedding/embedding_weights/part_0/Initializer/truncated_normal/TruncatedNormaliinput_layer/input_layer/occupation_embedding/embedding_weights/part_0/Initializer/truncated_normal/stddev*X
_classN
LJloc:@input_layer/input_layer/occupation_embedding/embedding_weights/part_0*
_output_shapes

:
*
T0
˝
binput_layer/input_layer/occupation_embedding/embedding_weights/part_0/Initializer/truncated_normalAddfinput_layer/input_layer/occupation_embedding/embedding_weights/part_0/Initializer/truncated_normal/mulginput_layer/input_layer/occupation_embedding/embedding_weights/part_0/Initializer/truncated_normal/mean*X
_classN
LJloc:@input_layer/input_layer/occupation_embedding/embedding_weights/part_0*
_output_shapes

:
*
T0
ď
Einput_layer/input_layer/occupation_embedding/embedding_weights/part_0
VariableV2*
_output_shapes

:
*
shape
:
*X
_classN
LJloc:@input_layer/input_layer/occupation_embedding/embedding_weights/part_0*
dtype0

Linput_layer/input_layer/occupation_embedding/embedding_weights/part_0/AssignAssignEinput_layer/input_layer/occupation_embedding/embedding_weights/part_0binput_layer/input_layer/occupation_embedding/embedding_weights/part_0/Initializer/truncated_normal*
T0*
_output_shapes

:
*X
_classN
LJloc:@input_layer/input_layer/occupation_embedding/embedding_weights/part_0
 
Jinput_layer/input_layer/occupation_embedding/embedding_weights/part_0/readIdentityEinput_layer/input_layer/occupation_embedding/embedding_weights/part_0*
_output_shapes

:
*X
_classN
LJloc:@input_layer/input_layer/occupation_embedding/embedding_weights/part_0*
T0

Uinput_layer/input_layer/occupation_embedding/occupation_embedding_weights/Slice/beginConst*
valueB: *
dtype0*
_output_shapes
:

Tinput_layer/input_layer/occupation_embedding/occupation_embedding_weights/Slice/sizeConst*
_output_shapes
:*
valueB:*
dtype0

Oinput_layer/input_layer/occupation_embedding/occupation_embedding_weights/SliceSliceHinput_layer/input_layer/occupation_embedding/to_sparse_input/dense_shapeUinput_layer/input_layer/occupation_embedding/occupation_embedding_weights/Slice/beginTinput_layer/input_layer/occupation_embedding/occupation_embedding_weights/Slice/size*
T0	*
Index0*
_output_shapes
:

Oinput_layer/input_layer/occupation_embedding/occupation_embedding_weights/ConstConst*
valueB: *
_output_shapes
:*
dtype0

Ninput_layer/input_layer/occupation_embedding/occupation_embedding_weights/ProdProdOinput_layer/input_layer/occupation_embedding/occupation_embedding_weights/SliceOinput_layer/input_layer/occupation_embedding/occupation_embedding_weights/Const*
_output_shapes
: *
T0	

Zinput_layer/input_layer/occupation_embedding/occupation_embedding_weights/GatherV2/indicesConst*
_output_shapes
: *
dtype0*
value	B :

Winput_layer/input_layer/occupation_embedding/occupation_embedding_weights/GatherV2/axisConst*
dtype0*
value	B : *
_output_shapes
: 
Ą
Rinput_layer/input_layer/occupation_embedding/occupation_embedding_weights/GatherV2GatherV2Hinput_layer/input_layer/occupation_embedding/to_sparse_input/dense_shapeZinput_layer/input_layer/occupation_embedding/occupation_embedding_weights/GatherV2/indicesWinput_layer/input_layer/occupation_embedding/occupation_embedding_weights/GatherV2/axis*
Taxis0*
_output_shapes
: *
Tparams0	*
Tindices0
Ş
Pinput_layer/input_layer/occupation_embedding/occupation_embedding_weights/Cast/xPackNinput_layer/input_layer/occupation_embedding/occupation_embedding_weights/ProdRinput_layer/input_layer/occupation_embedding/occupation_embedding_weights/GatherV2*
_output_shapes
:*
T0	*
N
ů
Winput_layer/input_layer/occupation_embedding/occupation_embedding_weights/SparseReshapeSparseReshapeDinput_layer/input_layer/occupation_embedding/to_sparse_input/indicesHinput_layer/input_layer/occupation_embedding/to_sparse_input/dense_shapePinput_layer/input_layer/occupation_embedding/occupation_embedding_weights/Cast/x*-
_output_shapes
:˙˙˙˙˙˙˙˙˙:
Ď
`input_layer/input_layer/occupation_embedding/occupation_embedding_weights/SparseReshape/IdentityIdentity3input_layer/input_layer/occupation_embedding/lookup*
T0	*#
_output_shapes
:˙˙˙˙˙˙˙˙˙

Xinput_layer/input_layer/occupation_embedding/occupation_embedding_weights/GreaterEqual/yConst*
value	B	 R *
_output_shapes
: *
dtype0	
Đ
Vinput_layer/input_layer/occupation_embedding/occupation_embedding_weights/GreaterEqualGreaterEqual`input_layer/input_layer/occupation_embedding/occupation_embedding_weights/SparseReshape/IdentityXinput_layer/input_layer/occupation_embedding/occupation_embedding_weights/GreaterEqual/y*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0	
Ů
Oinput_layer/input_layer/occupation_embedding/occupation_embedding_weights/WhereWhereVinput_layer/input_layer/occupation_embedding/occupation_embedding_weights/GreaterEqual*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ş
Winput_layer/input_layer/occupation_embedding/occupation_embedding_weights/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙
´
Qinput_layer/input_layer/occupation_embedding/occupation_embedding_weights/ReshapeReshapeOinput_layer/input_layer/occupation_embedding/occupation_embedding_weights/WhereWinput_layer/input_layer/occupation_embedding/occupation_embedding_weights/Reshape/shape*
T0	*#
_output_shapes
:˙˙˙˙˙˙˙˙˙

Yinput_layer/input_layer/occupation_embedding/occupation_embedding_weights/GatherV2_1/axisConst*
dtype0*
value	B : *
_output_shapes
: 
ź
Tinput_layer/input_layer/occupation_embedding/occupation_embedding_weights/GatherV2_1GatherV2Winput_layer/input_layer/occupation_embedding/occupation_embedding_weights/SparseReshapeQinput_layer/input_layer/occupation_embedding/occupation_embedding_weights/ReshapeYinput_layer/input_layer/occupation_embedding/occupation_embedding_weights/GatherV2_1/axis*
Tparams0	*
Taxis0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
Tindices0	

Yinput_layer/input_layer/occupation_embedding/occupation_embedding_weights/GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
value	B : 
Á
Tinput_layer/input_layer/occupation_embedding/occupation_embedding_weights/GatherV2_2GatherV2`input_layer/input_layer/occupation_embedding/occupation_embedding_weights/SparseReshape/IdentityQinput_layer/input_layer/occupation_embedding/occupation_embedding_weights/ReshapeYinput_layer/input_layer/occupation_embedding/occupation_embedding_weights/GatherV2_2/axis*
Taxis0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
Tindices0	*
Tparams0	
Ţ
Rinput_layer/input_layer/occupation_embedding/occupation_embedding_weights/IdentityIdentityYinput_layer/input_layer/occupation_embedding/occupation_embedding_weights/SparseReshape:1*
T0	*
_output_shapes
:
Ľ
cinput_layer/input_layer/occupation_embedding/occupation_embedding_weights/SparseFillEmptyRows/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R 
Ě
qinput_layer/input_layer/occupation_embedding/occupation_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowsSparseFillEmptyRowsTinput_layer/input_layer/occupation_embedding/occupation_embedding_weights/GatherV2_1Tinput_layer/input_layer/occupation_embedding/occupation_embedding_weights/GatherV2_2Rinput_layer/input_layer/occupation_embedding/occupation_embedding_weights/Identitycinput_layer/input_layer/occupation_embedding/occupation_embedding_weights/SparseFillEmptyRows/Const*T
_output_shapesB
@:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0	
Ć
uinput_layer/input_layer/occupation_embedding/occupation_embedding_weights/embedding_lookup_sparse/strided_slice/stackConst*
valueB"        *
dtype0*
_output_shapes
:
Č
winput_layer/input_layer/occupation_embedding/occupation_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB"       
Č
winput_layer/input_layer/occupation_embedding/occupation_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2Const*
_output_shapes
:*
valueB"      *
dtype0
Đ
oinput_layer/input_layer/occupation_embedding/occupation_embedding_weights/embedding_lookup_sparse/strided_sliceStridedSliceqinput_layer/input_layer/occupation_embedding/occupation_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowsuinput_layer/input_layer/occupation_embedding/occupation_embedding_weights/embedding_lookup_sparse/strided_slice/stackwinput_layer/input_layer/occupation_embedding/occupation_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1winput_layer/input_layer/occupation_embedding/occupation_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2*
T0	*
shrink_axis_mask*

begin_mask*
end_mask*
Index0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙

finput_layer/input_layer/occupation_embedding/occupation_embedding_weights/embedding_lookup_sparse/CastCastoinput_layer/input_layer/occupation_embedding/occupation_embedding_weights/embedding_lookup_sparse/strided_slice*

DstT0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*

SrcT0	
¤
hinput_layer/input_layer/occupation_embedding/occupation_embedding_weights/embedding_lookup_sparse/UniqueUniquesinput_layer/input_layer/occupation_embedding/occupation_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0	

winput_layer/input_layer/occupation_embedding/occupation_embedding_weights/embedding_lookup_sparse/embedding_lookup/axisConst*
value	B : *
_output_shapes
: *
dtype0*X
_classN
LJloc:@input_layer/input_layer/occupation_embedding/embedding_weights/part_0
Ü
rinput_layer/input_layer/occupation_embedding/occupation_embedding_weights/embedding_lookup_sparse/embedding_lookupGatherV2Jinput_layer/input_layer/occupation_embedding/embedding_weights/part_0/readhinput_layer/input_layer/occupation_embedding/occupation_embedding_weights/embedding_lookup_sparse/Uniquewinput_layer/input_layer/occupation_embedding/occupation_embedding_weights/embedding_lookup_sparse/embedding_lookup/axis*
Tparams0*X
_classN
LJloc:@input_layer/input_layer/occupation_embedding/embedding_weights/part_0*
Taxis0*
Tindices0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
­
{input_layer/input_layer/occupation_embedding/occupation_embedding_weights/embedding_lookup_sparse/embedding_lookup/IdentityIdentityrinput_layer/input_layer/occupation_embedding/occupation_embedding_weights/embedding_lookup_sparse/embedding_lookup*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
ů
ainput_layer/input_layer/occupation_embedding/occupation_embedding_weights/embedding_lookup_sparseSparseSegmentMean{input_layer/input_layer/occupation_embedding/occupation_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identityjinput_layer/input_layer/occupation_embedding/occupation_embedding_weights/embedding_lookup_sparse/Unique:1finput_layer/input_layer/occupation_embedding/occupation_embedding_weights/embedding_lookup_sparse/Cast*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ş
Yinput_layer/input_layer/occupation_embedding/occupation_embedding_weights/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   
ŕ
Sinput_layer/input_layer/occupation_embedding/occupation_embedding_weights/Reshape_1Reshapesinput_layer/input_layer/occupation_embedding/occupation_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:2Yinput_layer/input_layer/occupation_embedding/occupation_embedding_weights/Reshape_1/shape*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

ŕ
Oinput_layer/input_layer/occupation_embedding/occupation_embedding_weights/ShapeShapeainput_layer/input_layer/occupation_embedding/occupation_embedding_weights/embedding_lookup_sparse*
_output_shapes
:*
T0
§
]input_layer/input_layer/occupation_embedding/occupation_embedding_weights/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB:
Š
_input_layer/input_layer/occupation_embedding/occupation_embedding_weights/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
Š
_input_layer/input_layer/occupation_embedding/occupation_embedding_weights/strided_slice/stack_2Const*
valueB:*
_output_shapes
:*
dtype0

Winput_layer/input_layer/occupation_embedding/occupation_embedding_weights/strided_sliceStridedSliceOinput_layer/input_layer/occupation_embedding/occupation_embedding_weights/Shape]input_layer/input_layer/occupation_embedding/occupation_embedding_weights/strided_slice/stack_input_layer/input_layer/occupation_embedding/occupation_embedding_weights/strided_slice/stack_1_input_layer/input_layer/occupation_embedding/occupation_embedding_weights/strided_slice/stack_2*
Index0*
_output_shapes
: *
shrink_axis_mask*
T0

Qinput_layer/input_layer/occupation_embedding/occupation_embedding_weights/stack/0Const*
value	B :*
_output_shapes
: *
dtype0
ą
Oinput_layer/input_layer/occupation_embedding/occupation_embedding_weights/stackPackQinput_layer/input_layer/occupation_embedding/occupation_embedding_weights/stack/0Winput_layer/input_layer/occupation_embedding/occupation_embedding_weights/strided_slice*
T0*
N*
_output_shapes
:
ˇ
Ninput_layer/input_layer/occupation_embedding/occupation_embedding_weights/TileTileSinput_layer/input_layer/occupation_embedding/occupation_embedding_weights/Reshape_1Oinput_layer/input_layer/occupation_embedding/occupation_embedding_weights/stack*
T0
*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
ö
Tinput_layer/input_layer/occupation_embedding/occupation_embedding_weights/zeros_like	ZerosLikeainput_layer/input_layer/occupation_embedding/occupation_embedding_weights/embedding_lookup_sparse*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

Iinput_layer/input_layer/occupation_embedding/occupation_embedding_weightsSelectNinput_layer/input_layer/occupation_embedding/occupation_embedding_weights/TileTinput_layer/input_layer/occupation_embedding/occupation_embedding_weights/zeros_likeainput_layer/input_layer/occupation_embedding/occupation_embedding_weights/embedding_lookup_sparse*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ö
Pinput_layer/input_layer/occupation_embedding/occupation_embedding_weights/Cast_1CastHinput_layer/input_layer/occupation_embedding/to_sparse_input/dense_shape*

DstT0*

SrcT0	*
_output_shapes
:
Ą
Winput_layer/input_layer/occupation_embedding/occupation_embedding_weights/Slice_1/beginConst*
dtype0*
valueB: *
_output_shapes
:
 
Vinput_layer/input_layer/occupation_embedding/occupation_embedding_weights/Slice_1/sizeConst*
_output_shapes
:*
valueB:*
dtype0

Qinput_layer/input_layer/occupation_embedding/occupation_embedding_weights/Slice_1SlicePinput_layer/input_layer/occupation_embedding/occupation_embedding_weights/Cast_1Winput_layer/input_layer/occupation_embedding/occupation_embedding_weights/Slice_1/beginVinput_layer/input_layer/occupation_embedding/occupation_embedding_weights/Slice_1/size*
T0*
_output_shapes
:*
Index0
Ę
Qinput_layer/input_layer/occupation_embedding/occupation_embedding_weights/Shape_1ShapeIinput_layer/input_layer/occupation_embedding/occupation_embedding_weights*
T0*
_output_shapes
:
Ą
Winput_layer/input_layer/occupation_embedding/occupation_embedding_weights/Slice_2/beginConst*
dtype0*
valueB:*
_output_shapes
:
Š
Vinput_layer/input_layer/occupation_embedding/occupation_embedding_weights/Slice_2/sizeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:

Qinput_layer/input_layer/occupation_embedding/occupation_embedding_weights/Slice_2SliceQinput_layer/input_layer/occupation_embedding/occupation_embedding_weights/Shape_1Winput_layer/input_layer/occupation_embedding/occupation_embedding_weights/Slice_2/beginVinput_layer/input_layer/occupation_embedding/occupation_embedding_weights/Slice_2/size*
Index0*
_output_shapes
:*
T0

Uinput_layer/input_layer/occupation_embedding/occupation_embedding_weights/concat/axisConst*
_output_shapes
: *
value	B : *
dtype0

Pinput_layer/input_layer/occupation_embedding/occupation_embedding_weights/concatConcatV2Qinput_layer/input_layer/occupation_embedding/occupation_embedding_weights/Slice_1Qinput_layer/input_layer/occupation_embedding/occupation_embedding_weights/Slice_2Uinput_layer/input_layer/occupation_embedding/occupation_embedding_weights/concat/axis*
T0*
_output_shapes
:*
N
­
Sinput_layer/input_layer/occupation_embedding/occupation_embedding_weights/Reshape_2ReshapeIinput_layer/input_layer/occupation_embedding/occupation_embedding_weightsPinput_layer/input_layer/occupation_embedding/occupation_embedding_weights/concat*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
ľ
2input_layer/input_layer/occupation_embedding/ShapeShapeSinput_layer/input_layer/occupation_embedding/occupation_embedding_weights/Reshape_2*
_output_shapes
:*
T0

@input_layer/input_layer/occupation_embedding/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB: 

Binput_layer/input_layer/occupation_embedding/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:

Binput_layer/input_layer/occupation_embedding/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:

:input_layer/input_layer/occupation_embedding/strided_sliceStridedSlice2input_layer/input_layer/occupation_embedding/Shape@input_layer/input_layer/occupation_embedding/strided_slice/stackBinput_layer/input_layer/occupation_embedding/strided_slice/stack_1Binput_layer/input_layer/occupation_embedding/strided_slice/stack_2*
_output_shapes
: *
shrink_axis_mask*
Index0*
T0
~
<input_layer/input_layer/occupation_embedding/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
ę
:input_layer/input_layer/occupation_embedding/Reshape/shapePack:input_layer/input_layer/occupation_embedding/strided_slice<input_layer/input_layer/occupation_embedding/Reshape/shape/1*
T0*
_output_shapes
:*
N

4input_layer/input_layer/occupation_embedding/ReshapeReshapeSinput_layer/input_layer/occupation_embedding/occupation_embedding_weights/Reshape_2:input_layer/input_layer/occupation_embedding/Reshape/shape*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

=input_layer/input_layer/pvalue_level_embedding/ExpandDims/dimConst*
_output_shapes
: *
valueB :
˙˙˙˙˙˙˙˙˙*
dtype0
Č
9input_layer/input_layer/pvalue_level_embedding/ExpandDims
ExpandDimsPlaceholder_13=input_layer/input_layer/pvalue_level_embedding/ExpandDims/dim*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

Minput_layer/input_layer/pvalue_level_embedding/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
valueB B *
dtype0

Ginput_layer/input_layer/pvalue_level_embedding/to_sparse_input/NotEqualNotEqual9input_layer/input_layer/pvalue_level_embedding/ExpandDimsMinput_layer/input_layer/pvalue_level_embedding/to_sparse_input/ignore_value/x*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Á
Finput_layer/input_layer/pvalue_level_embedding/to_sparse_input/indicesWhereGinput_layer/input_layer/pvalue_level_embedding/to_sparse_input/NotEqual*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

Einput_layer/input_layer/pvalue_level_embedding/to_sparse_input/valuesGatherNd9input_layer/input_layer/pvalue_level_embedding/ExpandDimsFinput_layer/input_layer/pvalue_level_embedding/to_sparse_input/indices*
Tindices0	*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
Tparams0
Ă
Jinput_layer/input_layer/pvalue_level_embedding/to_sparse_input/dense_shapeShape9input_layer/input_layer/pvalue_level_embedding/ExpandDims*
out_type0	*
T0*
_output_shapes
:
Î
5input_layer/input_layer/pvalue_level_embedding/lookupStringToHashBucketFastEinput_layer/input_layer/pvalue_level_embedding/to_sparse_input/values*
num_buckets
*#
_output_shapes
:˙˙˙˙˙˙˙˙˙

jinput_layer/input_layer/pvalue_level_embedding/embedding_weights/part_0/Initializer/truncated_normal/shapeConst*Z
_classP
NLloc:@input_layer/input_layer/pvalue_level_embedding/embedding_weights/part_0*
valueB"
      *
_output_shapes
:*
dtype0

iinput_layer/input_layer/pvalue_level_embedding/embedding_weights/part_0/Initializer/truncated_normal/meanConst*
dtype0*Z
_classP
NLloc:@input_layer/input_layer/pvalue_level_embedding/embedding_weights/part_0*
valueB
 *    *
_output_shapes
: 

kinput_layer/input_layer/pvalue_level_embedding/embedding_weights/part_0/Initializer/truncated_normal/stddevConst*
valueB
 *  >*
dtype0*
_output_shapes
: *Z
_classP
NLloc:@input_layer/input_layer/pvalue_level_embedding/embedding_weights/part_0

tinput_layer/input_layer/pvalue_level_embedding/embedding_weights/part_0/Initializer/truncated_normal/TruncatedNormalTruncatedNormaljinput_layer/input_layer/pvalue_level_embedding/embedding_weights/part_0/Initializer/truncated_normal/shape*
dtype0*
T0*
_output_shapes

:
*Z
_classP
NLloc:@input_layer/input_layer/pvalue_level_embedding/embedding_weights/part_0
×
hinput_layer/input_layer/pvalue_level_embedding/embedding_weights/part_0/Initializer/truncated_normal/mulMultinput_layer/input_layer/pvalue_level_embedding/embedding_weights/part_0/Initializer/truncated_normal/TruncatedNormalkinput_layer/input_layer/pvalue_level_embedding/embedding_weights/part_0/Initializer/truncated_normal/stddev*
T0*
_output_shapes

:
*Z
_classP
NLloc:@input_layer/input_layer/pvalue_level_embedding/embedding_weights/part_0
Ĺ
dinput_layer/input_layer/pvalue_level_embedding/embedding_weights/part_0/Initializer/truncated_normalAddhinput_layer/input_layer/pvalue_level_embedding/embedding_weights/part_0/Initializer/truncated_normal/muliinput_layer/input_layer/pvalue_level_embedding/embedding_weights/part_0/Initializer/truncated_normal/mean*
T0*Z
_classP
NLloc:@input_layer/input_layer/pvalue_level_embedding/embedding_weights/part_0*
_output_shapes

:

ó
Ginput_layer/input_layer/pvalue_level_embedding/embedding_weights/part_0
VariableV2*
_output_shapes

:
*
shape
:
*Z
_classP
NLloc:@input_layer/input_layer/pvalue_level_embedding/embedding_weights/part_0*
dtype0

Ninput_layer/input_layer/pvalue_level_embedding/embedding_weights/part_0/AssignAssignGinput_layer/input_layer/pvalue_level_embedding/embedding_weights/part_0dinput_layer/input_layer/pvalue_level_embedding/embedding_weights/part_0/Initializer/truncated_normal*
T0*
_output_shapes

:
*Z
_classP
NLloc:@input_layer/input_layer/pvalue_level_embedding/embedding_weights/part_0
Ś
Linput_layer/input_layer/pvalue_level_embedding/embedding_weights/part_0/readIdentityGinput_layer/input_layer/pvalue_level_embedding/embedding_weights/part_0*Z
_classP
NLloc:@input_layer/input_layer/pvalue_level_embedding/embedding_weights/part_0*
T0*
_output_shapes

:

Ł
Yinput_layer/input_layer/pvalue_level_embedding/pvalue_level_embedding_weights/Slice/beginConst*
dtype0*
_output_shapes
:*
valueB: 
˘
Xinput_layer/input_layer/pvalue_level_embedding/pvalue_level_embedding_weights/Slice/sizeConst*
_output_shapes
:*
valueB:*
dtype0

Sinput_layer/input_layer/pvalue_level_embedding/pvalue_level_embedding_weights/SliceSliceJinput_layer/input_layer/pvalue_level_embedding/to_sparse_input/dense_shapeYinput_layer/input_layer/pvalue_level_embedding/pvalue_level_embedding_weights/Slice/beginXinput_layer/input_layer/pvalue_level_embedding/pvalue_level_embedding_weights/Slice/size*
_output_shapes
:*
Index0*
T0	

Sinput_layer/input_layer/pvalue_level_embedding/pvalue_level_embedding_weights/ConstConst*
dtype0*
_output_shapes
:*
valueB: 
Ľ
Rinput_layer/input_layer/pvalue_level_embedding/pvalue_level_embedding_weights/ProdProdSinput_layer/input_layer/pvalue_level_embedding/pvalue_level_embedding_weights/SliceSinput_layer/input_layer/pvalue_level_embedding/pvalue_level_embedding_weights/Const*
_output_shapes
: *
T0	
 
^input_layer/input_layer/pvalue_level_embedding/pvalue_level_embedding_weights/GatherV2/indicesConst*
value	B :*
dtype0*
_output_shapes
: 

[input_layer/input_layer/pvalue_level_embedding/pvalue_level_embedding_weights/GatherV2/axisConst*
value	B : *
dtype0*
_output_shapes
: 
Ż
Vinput_layer/input_layer/pvalue_level_embedding/pvalue_level_embedding_weights/GatherV2GatherV2Jinput_layer/input_layer/pvalue_level_embedding/to_sparse_input/dense_shape^input_layer/input_layer/pvalue_level_embedding/pvalue_level_embedding_weights/GatherV2/indices[input_layer/input_layer/pvalue_level_embedding/pvalue_level_embedding_weights/GatherV2/axis*
_output_shapes
: *
Tparams0	*
Tindices0*
Taxis0
ś
Tinput_layer/input_layer/pvalue_level_embedding/pvalue_level_embedding_weights/Cast/xPackRinput_layer/input_layer/pvalue_level_embedding/pvalue_level_embedding_weights/ProdVinput_layer/input_layer/pvalue_level_embedding/pvalue_level_embedding_weights/GatherV2*
N*
_output_shapes
:*
T0	

[input_layer/input_layer/pvalue_level_embedding/pvalue_level_embedding_weights/SparseReshapeSparseReshapeFinput_layer/input_layer/pvalue_level_embedding/to_sparse_input/indicesJinput_layer/input_layer/pvalue_level_embedding/to_sparse_input/dense_shapeTinput_layer/input_layer/pvalue_level_embedding/pvalue_level_embedding_weights/Cast/x*-
_output_shapes
:˙˙˙˙˙˙˙˙˙:
Ő
dinput_layer/input_layer/pvalue_level_embedding/pvalue_level_embedding_weights/SparseReshape/IdentityIdentity5input_layer/input_layer/pvalue_level_embedding/lookup*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0	

\input_layer/input_layer/pvalue_level_embedding/pvalue_level_embedding_weights/GreaterEqual/yConst*
dtype0	*
_output_shapes
: *
value	B	 R 
Ü
Zinput_layer/input_layer/pvalue_level_embedding/pvalue_level_embedding_weights/GreaterEqualGreaterEqualdinput_layer/input_layer/pvalue_level_embedding/pvalue_level_embedding_weights/SparseReshape/Identity\input_layer/input_layer/pvalue_level_embedding/pvalue_level_embedding_weights/GreaterEqual/y*
T0	*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
á
Sinput_layer/input_layer/pvalue_level_embedding/pvalue_level_embedding_weights/WhereWhereZinput_layer/input_layer/pvalue_level_embedding/pvalue_level_embedding_weights/GreaterEqual*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ž
[input_layer/input_layer/pvalue_level_embedding/pvalue_level_embedding_weights/Reshape/shapeConst*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙*
_output_shapes
:
Ŕ
Uinput_layer/input_layer/pvalue_level_embedding/pvalue_level_embedding_weights/ReshapeReshapeSinput_layer/input_layer/pvalue_level_embedding/pvalue_level_embedding_weights/Where[input_layer/input_layer/pvalue_level_embedding/pvalue_level_embedding_weights/Reshape/shape*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0	

]input_layer/input_layer/pvalue_level_embedding/pvalue_level_embedding_weights/GatherV2_1/axisConst*
value	B : *
_output_shapes
: *
dtype0
Ě
Xinput_layer/input_layer/pvalue_level_embedding/pvalue_level_embedding_weights/GatherV2_1GatherV2[input_layer/input_layer/pvalue_level_embedding/pvalue_level_embedding_weights/SparseReshapeUinput_layer/input_layer/pvalue_level_embedding/pvalue_level_embedding_weights/Reshape]input_layer/input_layer/pvalue_level_embedding/pvalue_level_embedding_weights/GatherV2_1/axis*
Taxis0*
Tparams0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
Tindices0	

]input_layer/input_layer/pvalue_level_embedding/pvalue_level_embedding_weights/GatherV2_2/axisConst*
_output_shapes
: *
value	B : *
dtype0
Ń
Xinput_layer/input_layer/pvalue_level_embedding/pvalue_level_embedding_weights/GatherV2_2GatherV2dinput_layer/input_layer/pvalue_level_embedding/pvalue_level_embedding_weights/SparseReshape/IdentityUinput_layer/input_layer/pvalue_level_embedding/pvalue_level_embedding_weights/Reshape]input_layer/input_layer/pvalue_level_embedding/pvalue_level_embedding_weights/GatherV2_2/axis*
Tparams0	*
Taxis0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
Tindices0	
ć
Vinput_layer/input_layer/pvalue_level_embedding/pvalue_level_embedding_weights/IdentityIdentity]input_layer/input_layer/pvalue_level_embedding/pvalue_level_embedding_weights/SparseReshape:1*
_output_shapes
:*
T0	
Š
ginput_layer/input_layer/pvalue_level_embedding/pvalue_level_embedding_weights/SparseFillEmptyRows/ConstConst*
value	B	 R *
dtype0	*
_output_shapes
: 
ŕ
uinput_layer/input_layer/pvalue_level_embedding/pvalue_level_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowsSparseFillEmptyRowsXinput_layer/input_layer/pvalue_level_embedding/pvalue_level_embedding_weights/GatherV2_1Xinput_layer/input_layer/pvalue_level_embedding/pvalue_level_embedding_weights/GatherV2_2Vinput_layer/input_layer/pvalue_level_embedding/pvalue_level_embedding_weights/Identityginput_layer/input_layer/pvalue_level_embedding/pvalue_level_embedding_weights/SparseFillEmptyRows/Const*T
_output_shapesB
@:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0	
Ę
yinput_layer/input_layer/pvalue_level_embedding/pvalue_level_embedding_weights/embedding_lookup_sparse/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB"        
Ě
{input_layer/input_layer/pvalue_level_embedding/pvalue_level_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1Const*
_output_shapes
:*
valueB"       *
dtype0
Ě
{input_layer/input_layer/pvalue_level_embedding/pvalue_level_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:
ä
sinput_layer/input_layer/pvalue_level_embedding/pvalue_level_embedding_weights/embedding_lookup_sparse/strided_sliceStridedSliceuinput_layer/input_layer/pvalue_level_embedding/pvalue_level_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowsyinput_layer/input_layer/pvalue_level_embedding/pvalue_level_embedding_weights/embedding_lookup_sparse/strided_slice/stack{input_layer/input_layer/pvalue_level_embedding/pvalue_level_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1{input_layer/input_layer/pvalue_level_embedding/pvalue_level_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2*

begin_mask*
shrink_axis_mask*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
Index0*
end_mask*
T0	
¤
jinput_layer/input_layer/pvalue_level_embedding/pvalue_level_embedding_weights/embedding_lookup_sparse/CastCastsinput_layer/input_layer/pvalue_level_embedding/pvalue_level_embedding_weights/embedding_lookup_sparse/strided_slice*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*

SrcT0	*

DstT0
Ź
linput_layer/input_layer/pvalue_level_embedding/pvalue_level_embedding_weights/embedding_lookup_sparse/UniqueUniquewinput_layer/input_layer/pvalue_level_embedding/pvalue_level_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:1*
T0	*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙

{input_layer/input_layer/pvalue_level_embedding/pvalue_level_embedding_weights/embedding_lookup_sparse/embedding_lookup/axisConst*
_output_shapes
: *
value	B : *
dtype0*Z
_classP
NLloc:@input_layer/input_layer/pvalue_level_embedding/embedding_weights/part_0
ě
vinput_layer/input_layer/pvalue_level_embedding/pvalue_level_embedding_weights/embedding_lookup_sparse/embedding_lookupGatherV2Linput_layer/input_layer/pvalue_level_embedding/embedding_weights/part_0/readlinput_layer/input_layer/pvalue_level_embedding/pvalue_level_embedding_weights/embedding_lookup_sparse/Unique{input_layer/input_layer/pvalue_level_embedding/pvalue_level_embedding_weights/embedding_lookup_sparse/embedding_lookup/axis*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
Tindices0	*
Taxis0*Z
_classP
NLloc:@input_layer/input_layer/pvalue_level_embedding/embedding_weights/part_0*
Tparams0
ľ
input_layer/input_layer/pvalue_level_embedding/pvalue_level_embedding_weights/embedding_lookup_sparse/embedding_lookup/IdentityIdentityvinput_layer/input_layer/pvalue_level_embedding/pvalue_level_embedding_weights/embedding_lookup_sparse/embedding_lookup*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

einput_layer/input_layer/pvalue_level_embedding/pvalue_level_embedding_weights/embedding_lookup_sparseSparseSegmentMeaninput_layer/input_layer/pvalue_level_embedding/pvalue_level_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identityninput_layer/input_layer/pvalue_level_embedding/pvalue_level_embedding_weights/embedding_lookup_sparse/Unique:1jinput_layer/input_layer/pvalue_level_embedding/pvalue_level_embedding_weights/embedding_lookup_sparse/Cast*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ž
]input_layer/input_layer/pvalue_level_embedding/pvalue_level_embedding_weights/Reshape_1/shapeConst*
valueB"˙˙˙˙   *
dtype0*
_output_shapes
:
ě
Winput_layer/input_layer/pvalue_level_embedding/pvalue_level_embedding_weights/Reshape_1Reshapewinput_layer/input_layer/pvalue_level_embedding/pvalue_level_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:2]input_layer/input_layer/pvalue_level_embedding/pvalue_level_embedding_weights/Reshape_1/shape*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

č
Sinput_layer/input_layer/pvalue_level_embedding/pvalue_level_embedding_weights/ShapeShapeeinput_layer/input_layer/pvalue_level_embedding/pvalue_level_embedding_weights/embedding_lookup_sparse*
_output_shapes
:*
T0
Ť
ainput_layer/input_layer/pvalue_level_embedding/pvalue_level_embedding_weights/strided_slice/stackConst*
valueB:*
dtype0*
_output_shapes
:
­
cinput_layer/input_layer/pvalue_level_embedding/pvalue_level_embedding_weights/strided_slice/stack_1Const*
_output_shapes
:*
valueB:*
dtype0
­
cinput_layer/input_layer/pvalue_level_embedding/pvalue_level_embedding_weights/strided_slice/stack_2Const*
_output_shapes
:*
valueB:*
dtype0
ł
[input_layer/input_layer/pvalue_level_embedding/pvalue_level_embedding_weights/strided_sliceStridedSliceSinput_layer/input_layer/pvalue_level_embedding/pvalue_level_embedding_weights/Shapeainput_layer/input_layer/pvalue_level_embedding/pvalue_level_embedding_weights/strided_slice/stackcinput_layer/input_layer/pvalue_level_embedding/pvalue_level_embedding_weights/strided_slice/stack_1cinput_layer/input_layer/pvalue_level_embedding/pvalue_level_embedding_weights/strided_slice/stack_2*
_output_shapes
: *
shrink_axis_mask*
Index0*
T0

Uinput_layer/input_layer/pvalue_level_embedding/pvalue_level_embedding_weights/stack/0Const*
value	B :*
_output_shapes
: *
dtype0
˝
Sinput_layer/input_layer/pvalue_level_embedding/pvalue_level_embedding_weights/stackPackUinput_layer/input_layer/pvalue_level_embedding/pvalue_level_embedding_weights/stack/0[input_layer/input_layer/pvalue_level_embedding/pvalue_level_embedding_weights/strided_slice*
N*
T0*
_output_shapes
:
Ă
Rinput_layer/input_layer/pvalue_level_embedding/pvalue_level_embedding_weights/TileTileWinput_layer/input_layer/pvalue_level_embedding/pvalue_level_embedding_weights/Reshape_1Sinput_layer/input_layer/pvalue_level_embedding/pvalue_level_embedding_weights/stack*
T0
*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
ţ
Xinput_layer/input_layer/pvalue_level_embedding/pvalue_level_embedding_weights/zeros_like	ZerosLikeeinput_layer/input_layer/pvalue_level_embedding/pvalue_level_embedding_weights/embedding_lookup_sparse*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

Minput_layer/input_layer/pvalue_level_embedding/pvalue_level_embedding_weightsSelectRinput_layer/input_layer/pvalue_level_embedding/pvalue_level_embedding_weights/TileXinput_layer/input_layer/pvalue_level_embedding/pvalue_level_embedding_weights/zeros_likeeinput_layer/input_layer/pvalue_level_embedding/pvalue_level_embedding_weights/embedding_lookup_sparse*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ü
Tinput_layer/input_layer/pvalue_level_embedding/pvalue_level_embedding_weights/Cast_1CastJinput_layer/input_layer/pvalue_level_embedding/to_sparse_input/dense_shape*
_output_shapes
:*

DstT0*

SrcT0	
Ľ
[input_layer/input_layer/pvalue_level_embedding/pvalue_level_embedding_weights/Slice_1/beginConst*
_output_shapes
:*
valueB: *
dtype0
¤
Zinput_layer/input_layer/pvalue_level_embedding/pvalue_level_embedding_weights/Slice_1/sizeConst*
dtype0*
valueB:*
_output_shapes
:

Uinput_layer/input_layer/pvalue_level_embedding/pvalue_level_embedding_weights/Slice_1SliceTinput_layer/input_layer/pvalue_level_embedding/pvalue_level_embedding_weights/Cast_1[input_layer/input_layer/pvalue_level_embedding/pvalue_level_embedding_weights/Slice_1/beginZinput_layer/input_layer/pvalue_level_embedding/pvalue_level_embedding_weights/Slice_1/size*
T0*
Index0*
_output_shapes
:
Ň
Uinput_layer/input_layer/pvalue_level_embedding/pvalue_level_embedding_weights/Shape_1ShapeMinput_layer/input_layer/pvalue_level_embedding/pvalue_level_embedding_weights*
T0*
_output_shapes
:
Ľ
[input_layer/input_layer/pvalue_level_embedding/pvalue_level_embedding_weights/Slice_2/beginConst*
dtype0*
valueB:*
_output_shapes
:
­
Zinput_layer/input_layer/pvalue_level_embedding/pvalue_level_embedding_weights/Slice_2/sizeConst*
dtype0*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙
 
Uinput_layer/input_layer/pvalue_level_embedding/pvalue_level_embedding_weights/Slice_2SliceUinput_layer/input_layer/pvalue_level_embedding/pvalue_level_embedding_weights/Shape_1[input_layer/input_layer/pvalue_level_embedding/pvalue_level_embedding_weights/Slice_2/beginZinput_layer/input_layer/pvalue_level_embedding/pvalue_level_embedding_weights/Slice_2/size*
Index0*
T0*
_output_shapes
:

Yinput_layer/input_layer/pvalue_level_embedding/pvalue_level_embedding_weights/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 

Tinput_layer/input_layer/pvalue_level_embedding/pvalue_level_embedding_weights/concatConcatV2Uinput_layer/input_layer/pvalue_level_embedding/pvalue_level_embedding_weights/Slice_1Uinput_layer/input_layer/pvalue_level_embedding/pvalue_level_embedding_weights/Slice_2Yinput_layer/input_layer/pvalue_level_embedding/pvalue_level_embedding_weights/concat/axis*
N*
_output_shapes
:*
T0
š
Winput_layer/input_layer/pvalue_level_embedding/pvalue_level_embedding_weights/Reshape_2ReshapeMinput_layer/input_layer/pvalue_level_embedding/pvalue_level_embedding_weightsTinput_layer/input_layer/pvalue_level_embedding/pvalue_level_embedding_weights/concat*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
ť
4input_layer/input_layer/pvalue_level_embedding/ShapeShapeWinput_layer/input_layer/pvalue_level_embedding/pvalue_level_embedding_weights/Reshape_2*
_output_shapes
:*
T0

Binput_layer/input_layer/pvalue_level_embedding/strided_slice/stackConst*
_output_shapes
:*
valueB: *
dtype0

Dinput_layer/input_layer/pvalue_level_embedding/strided_slice/stack_1Const*
dtype0*
valueB:*
_output_shapes
:

Dinput_layer/input_layer/pvalue_level_embedding/strided_slice/stack_2Const*
dtype0*
valueB:*
_output_shapes
:

<input_layer/input_layer/pvalue_level_embedding/strided_sliceStridedSlice4input_layer/input_layer/pvalue_level_embedding/ShapeBinput_layer/input_layer/pvalue_level_embedding/strided_slice/stackDinput_layer/input_layer/pvalue_level_embedding/strided_slice/stack_1Dinput_layer/input_layer/pvalue_level_embedding/strided_slice/stack_2*
_output_shapes
: *
shrink_axis_mask*
Index0*
T0

>input_layer/input_layer/pvalue_level_embedding/Reshape/shape/1Const*
_output_shapes
: *
value	B :*
dtype0
đ
<input_layer/input_layer/pvalue_level_embedding/Reshape/shapePack<input_layer/input_layer/pvalue_level_embedding/strided_slice>input_layer/input_layer/pvalue_level_embedding/Reshape/shape/1*
T0*
_output_shapes
:*
N

6input_layer/input_layer/pvalue_level_embedding/ReshapeReshapeWinput_layer/input_layer/pvalue_level_embedding/pvalue_level_embedding_weights/Reshape_2<input_layer/input_layer/pvalue_level_embedding/Reshape/shape*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

?input_layer/input_layer/shopping_level_embedding/ExpandDims/dimConst*
_output_shapes
: *
valueB :
˙˙˙˙˙˙˙˙˙*
dtype0
Ě
;input_layer/input_layer/shopping_level_embedding/ExpandDims
ExpandDimsPlaceholder_14?input_layer/input_layer/shopping_level_embedding/ExpandDims/dim*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

Oinput_layer/input_layer/shopping_level_embedding/to_sparse_input/ignore_value/xConst*
dtype0*
valueB B *
_output_shapes
: 

Iinput_layer/input_layer/shopping_level_embedding/to_sparse_input/NotEqualNotEqual;input_layer/input_layer/shopping_level_embedding/ExpandDimsOinput_layer/input_layer/shopping_level_embedding/to_sparse_input/ignore_value/x*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ĺ
Hinput_layer/input_layer/shopping_level_embedding/to_sparse_input/indicesWhereIinput_layer/input_layer/shopping_level_embedding/to_sparse_input/NotEqual*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

Ginput_layer/input_layer/shopping_level_embedding/to_sparse_input/valuesGatherNd;input_layer/input_layer/shopping_level_embedding/ExpandDimsHinput_layer/input_layer/shopping_level_embedding/to_sparse_input/indices*
Tparams0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
Tindices0	
Ç
Linput_layer/input_layer/shopping_level_embedding/to_sparse_input/dense_shapeShape;input_layer/input_layer/shopping_level_embedding/ExpandDims*
_output_shapes
:*
T0*
out_type0	
Ň
7input_layer/input_layer/shopping_level_embedding/lookupStringToHashBucketFastGinput_layer/input_layer/shopping_level_embedding/to_sparse_input/values*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
num_buckets


linput_layer/input_layer/shopping_level_embedding/embedding_weights/part_0/Initializer/truncated_normal/shapeConst*
valueB"
      *
_output_shapes
:*\
_classR
PNloc:@input_layer/input_layer/shopping_level_embedding/embedding_weights/part_0*
dtype0

kinput_layer/input_layer/shopping_level_embedding/embedding_weights/part_0/Initializer/truncated_normal/meanConst*
_output_shapes
: *
valueB
 *    *\
_classR
PNloc:@input_layer/input_layer/shopping_level_embedding/embedding_weights/part_0*
dtype0

minput_layer/input_layer/shopping_level_embedding/embedding_weights/part_0/Initializer/truncated_normal/stddevConst*
_output_shapes
: *
valueB
 *  >*
dtype0*\
_classR
PNloc:@input_layer/input_layer/shopping_level_embedding/embedding_weights/part_0

vinput_layer/input_layer/shopping_level_embedding/embedding_weights/part_0/Initializer/truncated_normal/TruncatedNormalTruncatedNormallinput_layer/input_layer/shopping_level_embedding/embedding_weights/part_0/Initializer/truncated_normal/shape*\
_classR
PNloc:@input_layer/input_layer/shopping_level_embedding/embedding_weights/part_0*
_output_shapes

:
*
dtype0*
T0
ß
jinput_layer/input_layer/shopping_level_embedding/embedding_weights/part_0/Initializer/truncated_normal/mulMulvinput_layer/input_layer/shopping_level_embedding/embedding_weights/part_0/Initializer/truncated_normal/TruncatedNormalminput_layer/input_layer/shopping_level_embedding/embedding_weights/part_0/Initializer/truncated_normal/stddev*\
_classR
PNloc:@input_layer/input_layer/shopping_level_embedding/embedding_weights/part_0*
_output_shapes

:
*
T0
Í
finput_layer/input_layer/shopping_level_embedding/embedding_weights/part_0/Initializer/truncated_normalAddjinput_layer/input_layer/shopping_level_embedding/embedding_weights/part_0/Initializer/truncated_normal/mulkinput_layer/input_layer/shopping_level_embedding/embedding_weights/part_0/Initializer/truncated_normal/mean*\
_classR
PNloc:@input_layer/input_layer/shopping_level_embedding/embedding_weights/part_0*
T0*
_output_shapes

:

÷
Iinput_layer/input_layer/shopping_level_embedding/embedding_weights/part_0
VariableV2*
shape
:
*
dtype0*
_output_shapes

:
*\
_classR
PNloc:@input_layer/input_layer/shopping_level_embedding/embedding_weights/part_0

Pinput_layer/input_layer/shopping_level_embedding/embedding_weights/part_0/AssignAssignIinput_layer/input_layer/shopping_level_embedding/embedding_weights/part_0finput_layer/input_layer/shopping_level_embedding/embedding_weights/part_0/Initializer/truncated_normal*
_output_shapes

:
*
T0*\
_classR
PNloc:@input_layer/input_layer/shopping_level_embedding/embedding_weights/part_0
Ź
Ninput_layer/input_layer/shopping_level_embedding/embedding_weights/part_0/readIdentityIinput_layer/input_layer/shopping_level_embedding/embedding_weights/part_0*
T0*\
_classR
PNloc:@input_layer/input_layer/shopping_level_embedding/embedding_weights/part_0*
_output_shapes

:

§
]input_layer/input_layer/shopping_level_embedding/shopping_level_embedding_weights/Slice/beginConst*
valueB: *
_output_shapes
:*
dtype0
Ś
\input_layer/input_layer/shopping_level_embedding/shopping_level_embedding_weights/Slice/sizeConst*
valueB:*
_output_shapes
:*
dtype0

Winput_layer/input_layer/shopping_level_embedding/shopping_level_embedding_weights/SliceSliceLinput_layer/input_layer/shopping_level_embedding/to_sparse_input/dense_shape]input_layer/input_layer/shopping_level_embedding/shopping_level_embedding_weights/Slice/begin\input_layer/input_layer/shopping_level_embedding/shopping_level_embedding_weights/Slice/size*
T0	*
_output_shapes
:*
Index0
Ą
Winput_layer/input_layer/shopping_level_embedding/shopping_level_embedding_weights/ConstConst*
valueB: *
_output_shapes
:*
dtype0
ą
Vinput_layer/input_layer/shopping_level_embedding/shopping_level_embedding_weights/ProdProdWinput_layer/input_layer/shopping_level_embedding/shopping_level_embedding_weights/SliceWinput_layer/input_layer/shopping_level_embedding/shopping_level_embedding_weights/Const*
_output_shapes
: *
T0	
¤
binput_layer/input_layer/shopping_level_embedding/shopping_level_embedding_weights/GatherV2/indicesConst*
_output_shapes
: *
value	B :*
dtype0
Ą
_input_layer/input_layer/shopping_level_embedding/shopping_level_embedding_weights/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 
˝
Zinput_layer/input_layer/shopping_level_embedding/shopping_level_embedding_weights/GatherV2GatherV2Linput_layer/input_layer/shopping_level_embedding/to_sparse_input/dense_shapebinput_layer/input_layer/shopping_level_embedding/shopping_level_embedding_weights/GatherV2/indices_input_layer/input_layer/shopping_level_embedding/shopping_level_embedding_weights/GatherV2/axis*
Tparams0	*
_output_shapes
: *
Taxis0*
Tindices0
Â
Xinput_layer/input_layer/shopping_level_embedding/shopping_level_embedding_weights/Cast/xPackVinput_layer/input_layer/shopping_level_embedding/shopping_level_embedding_weights/ProdZinput_layer/input_layer/shopping_level_embedding/shopping_level_embedding_weights/GatherV2*
_output_shapes
:*
N*
T0	

_input_layer/input_layer/shopping_level_embedding/shopping_level_embedding_weights/SparseReshapeSparseReshapeHinput_layer/input_layer/shopping_level_embedding/to_sparse_input/indicesLinput_layer/input_layer/shopping_level_embedding/to_sparse_input/dense_shapeXinput_layer/input_layer/shopping_level_embedding/shopping_level_embedding_weights/Cast/x*-
_output_shapes
:˙˙˙˙˙˙˙˙˙:
Ű
hinput_layer/input_layer/shopping_level_embedding/shopping_level_embedding_weights/SparseReshape/IdentityIdentity7input_layer/input_layer/shopping_level_embedding/lookup*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0	
˘
`input_layer/input_layer/shopping_level_embedding/shopping_level_embedding_weights/GreaterEqual/yConst*
_output_shapes
: *
value	B	 R *
dtype0	
č
^input_layer/input_layer/shopping_level_embedding/shopping_level_embedding_weights/GreaterEqualGreaterEqualhinput_layer/input_layer/shopping_level_embedding/shopping_level_embedding_weights/SparseReshape/Identity`input_layer/input_layer/shopping_level_embedding/shopping_level_embedding_weights/GreaterEqual/y*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0	
é
Winput_layer/input_layer/shopping_level_embedding/shopping_level_embedding_weights/WhereWhere^input_layer/input_layer/shopping_level_embedding/shopping_level_embedding_weights/GreaterEqual*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
˛
_input_layer/input_layer/shopping_level_embedding/shopping_level_embedding_weights/Reshape/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
_output_shapes
:*
dtype0
Ě
Yinput_layer/input_layer/shopping_level_embedding/shopping_level_embedding_weights/ReshapeReshapeWinput_layer/input_layer/shopping_level_embedding/shopping_level_embedding_weights/Where_input_layer/input_layer/shopping_level_embedding/shopping_level_embedding_weights/Reshape/shape*
T0	*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ł
ainput_layer/input_layer/shopping_level_embedding/shopping_level_embedding_weights/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
Ü
\input_layer/input_layer/shopping_level_embedding/shopping_level_embedding_weights/GatherV2_1GatherV2_input_layer/input_layer/shopping_level_embedding/shopping_level_embedding_weights/SparseReshapeYinput_layer/input_layer/shopping_level_embedding/shopping_level_embedding_weights/Reshapeainput_layer/input_layer/shopping_level_embedding/shopping_level_embedding_weights/GatherV2_1/axis*
Taxis0*
Tparams0	*
Tindices0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ł
ainput_layer/input_layer/shopping_level_embedding/shopping_level_embedding_weights/GatherV2_2/axisConst*
_output_shapes
: *
value	B : *
dtype0
á
\input_layer/input_layer/shopping_level_embedding/shopping_level_embedding_weights/GatherV2_2GatherV2hinput_layer/input_layer/shopping_level_embedding/shopping_level_embedding_weights/SparseReshape/IdentityYinput_layer/input_layer/shopping_level_embedding/shopping_level_embedding_weights/Reshapeainput_layer/input_layer/shopping_level_embedding/shopping_level_embedding_weights/GatherV2_2/axis*
Taxis0*
Tparams0	*
Tindices0	*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
î
Zinput_layer/input_layer/shopping_level_embedding/shopping_level_embedding_weights/IdentityIdentityainput_layer/input_layer/shopping_level_embedding/shopping_level_embedding_weights/SparseReshape:1*
T0	*
_output_shapes
:
­
kinput_layer/input_layer/shopping_level_embedding/shopping_level_embedding_weights/SparseFillEmptyRows/ConstConst*
_output_shapes
: *
value	B	 R *
dtype0	
ô
yinput_layer/input_layer/shopping_level_embedding/shopping_level_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowsSparseFillEmptyRows\input_layer/input_layer/shopping_level_embedding/shopping_level_embedding_weights/GatherV2_1\input_layer/input_layer/shopping_level_embedding/shopping_level_embedding_weights/GatherV2_2Zinput_layer/input_layer/shopping_level_embedding/shopping_level_embedding_weights/Identitykinput_layer/input_layer/shopping_level_embedding/shopping_level_embedding_weights/SparseFillEmptyRows/Const*
T0	*T
_output_shapesB
@:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
Î
}input_layer/input_layer/shopping_level_embedding/shopping_level_embedding_weights/embedding_lookup_sparse/strided_slice/stackConst*
dtype0*
valueB"        *
_output_shapes
:
Đ
input_layer/input_layer/shopping_level_embedding/shopping_level_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1Const*
dtype0*
valueB"       *
_output_shapes
:
Đ
input_layer/input_layer/shopping_level_embedding/shopping_level_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2Const*
_output_shapes
:*
valueB"      *
dtype0
ř
winput_layer/input_layer/shopping_level_embedding/shopping_level_embedding_weights/embedding_lookup_sparse/strided_sliceStridedSliceyinput_layer/input_layer/shopping_level_embedding/shopping_level_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows}input_layer/input_layer/shopping_level_embedding/shopping_level_embedding_weights/embedding_lookup_sparse/strided_slice/stackinput_layer/input_layer/shopping_level_embedding/shopping_level_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1input_layer/input_layer/shopping_level_embedding/shopping_level_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2*
end_mask*
T0	*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*

begin_mask*
Index0*
shrink_axis_mask
Ź
ninput_layer/input_layer/shopping_level_embedding/shopping_level_embedding_weights/embedding_lookup_sparse/CastCastwinput_layer/input_layer/shopping_level_embedding/shopping_level_embedding_weights/embedding_lookup_sparse/strided_slice*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*

SrcT0	*

DstT0
´
pinput_layer/input_layer/shopping_level_embedding/shopping_level_embedding_weights/embedding_lookup_sparse/UniqueUnique{input_layer/input_layer/shopping_level_embedding/shopping_level_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0	

input_layer/input_layer/shopping_level_embedding/shopping_level_embedding_weights/embedding_lookup_sparse/embedding_lookup/axisConst*
value	B : *
dtype0*
_output_shapes
: *\
_classR
PNloc:@input_layer/input_layer/shopping_level_embedding/embedding_weights/part_0
ü
zinput_layer/input_layer/shopping_level_embedding/shopping_level_embedding_weights/embedding_lookup_sparse/embedding_lookupGatherV2Ninput_layer/input_layer/shopping_level_embedding/embedding_weights/part_0/readpinput_layer/input_layer/shopping_level_embedding/shopping_level_embedding_weights/embedding_lookup_sparse/Uniqueinput_layer/input_layer/shopping_level_embedding/shopping_level_embedding_weights/embedding_lookup_sparse/embedding_lookup/axis*\
_classR
PNloc:@input_layer/input_layer/shopping_level_embedding/embedding_weights/part_0*
Taxis0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
Tparams0*
Tindices0	
ž
input_layer/input_layer/shopping_level_embedding/shopping_level_embedding_weights/embedding_lookup_sparse/embedding_lookup/IdentityIdentityzinput_layer/input_layer/shopping_level_embedding/shopping_level_embedding_weights/embedding_lookup_sparse/embedding_lookup*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

iinput_layer/input_layer/shopping_level_embedding/shopping_level_embedding_weights/embedding_lookup_sparseSparseSegmentMeaninput_layer/input_layer/shopping_level_embedding/shopping_level_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identityrinput_layer/input_layer/shopping_level_embedding/shopping_level_embedding_weights/embedding_lookup_sparse/Unique:1ninput_layer/input_layer/shopping_level_embedding/shopping_level_embedding_weights/embedding_lookup_sparse/Cast*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
˛
ainput_layer/input_layer/shopping_level_embedding/shopping_level_embedding_weights/Reshape_1/shapeConst*
_output_shapes
:*
valueB"˙˙˙˙   *
dtype0
ř
[input_layer/input_layer/shopping_level_embedding/shopping_level_embedding_weights/Reshape_1Reshape{input_layer/input_layer/shopping_level_embedding/shopping_level_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:2ainput_layer/input_layer/shopping_level_embedding/shopping_level_embedding_weights/Reshape_1/shape*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

đ
Winput_layer/input_layer/shopping_level_embedding/shopping_level_embedding_weights/ShapeShapeiinput_layer/input_layer/shopping_level_embedding/shopping_level_embedding_weights/embedding_lookup_sparse*
T0*
_output_shapes
:
Ż
einput_layer/input_layer/shopping_level_embedding/shopping_level_embedding_weights/strided_slice/stackConst*
valueB:*
dtype0*
_output_shapes
:
ą
ginput_layer/input_layer/shopping_level_embedding/shopping_level_embedding_weights/strided_slice/stack_1Const*
_output_shapes
:*
valueB:*
dtype0
ą
ginput_layer/input_layer/shopping_level_embedding/shopping_level_embedding_weights/strided_slice/stack_2Const*
valueB:*
_output_shapes
:*
dtype0
Ç
_input_layer/input_layer/shopping_level_embedding/shopping_level_embedding_weights/strided_sliceStridedSliceWinput_layer/input_layer/shopping_level_embedding/shopping_level_embedding_weights/Shapeeinput_layer/input_layer/shopping_level_embedding/shopping_level_embedding_weights/strided_slice/stackginput_layer/input_layer/shopping_level_embedding/shopping_level_embedding_weights/strided_slice/stack_1ginput_layer/input_layer/shopping_level_embedding/shopping_level_embedding_weights/strided_slice/stack_2*
Index0*
_output_shapes
: *
shrink_axis_mask*
T0

Yinput_layer/input_layer/shopping_level_embedding/shopping_level_embedding_weights/stack/0Const*
_output_shapes
: *
value	B :*
dtype0
É
Winput_layer/input_layer/shopping_level_embedding/shopping_level_embedding_weights/stackPackYinput_layer/input_layer/shopping_level_embedding/shopping_level_embedding_weights/stack/0_input_layer/input_layer/shopping_level_embedding/shopping_level_embedding_weights/strided_slice*
N*
_output_shapes
:*
T0
Ď
Vinput_layer/input_layer/shopping_level_embedding/shopping_level_embedding_weights/TileTile[input_layer/input_layer/shopping_level_embedding/shopping_level_embedding_weights/Reshape_1Winput_layer/input_layer/shopping_level_embedding/shopping_level_embedding_weights/stack*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
T0


\input_layer/input_layer/shopping_level_embedding/shopping_level_embedding_weights/zeros_like	ZerosLikeiinput_layer/input_layer/shopping_level_embedding/shopping_level_embedding_weights/embedding_lookup_sparse*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ž
Qinput_layer/input_layer/shopping_level_embedding/shopping_level_embedding_weightsSelectVinput_layer/input_layer/shopping_level_embedding/shopping_level_embedding_weights/Tile\input_layer/input_layer/shopping_level_embedding/shopping_level_embedding_weights/zeros_likeiinput_layer/input_layer/shopping_level_embedding/shopping_level_embedding_weights/embedding_lookup_sparse*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
â
Xinput_layer/input_layer/shopping_level_embedding/shopping_level_embedding_weights/Cast_1CastLinput_layer/input_layer/shopping_level_embedding/to_sparse_input/dense_shape*

DstT0*

SrcT0	*
_output_shapes
:
Š
_input_layer/input_layer/shopping_level_embedding/shopping_level_embedding_weights/Slice_1/beginConst*
dtype0*
_output_shapes
:*
valueB: 
¨
^input_layer/input_layer/shopping_level_embedding/shopping_level_embedding_weights/Slice_1/sizeConst*
valueB:*
_output_shapes
:*
dtype0
Ż
Yinput_layer/input_layer/shopping_level_embedding/shopping_level_embedding_weights/Slice_1SliceXinput_layer/input_layer/shopping_level_embedding/shopping_level_embedding_weights/Cast_1_input_layer/input_layer/shopping_level_embedding/shopping_level_embedding_weights/Slice_1/begin^input_layer/input_layer/shopping_level_embedding/shopping_level_embedding_weights/Slice_1/size*
_output_shapes
:*
Index0*
T0
Ú
Yinput_layer/input_layer/shopping_level_embedding/shopping_level_embedding_weights/Shape_1ShapeQinput_layer/input_layer/shopping_level_embedding/shopping_level_embedding_weights*
_output_shapes
:*
T0
Š
_input_layer/input_layer/shopping_level_embedding/shopping_level_embedding_weights/Slice_2/beginConst*
_output_shapes
:*
dtype0*
valueB:
ą
^input_layer/input_layer/shopping_level_embedding/shopping_level_embedding_weights/Slice_2/sizeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
_output_shapes
:*
dtype0
°
Yinput_layer/input_layer/shopping_level_embedding/shopping_level_embedding_weights/Slice_2SliceYinput_layer/input_layer/shopping_level_embedding/shopping_level_embedding_weights/Shape_1_input_layer/input_layer/shopping_level_embedding/shopping_level_embedding_weights/Slice_2/begin^input_layer/input_layer/shopping_level_embedding/shopping_level_embedding_weights/Slice_2/size*
Index0*
T0*
_output_shapes
:

]input_layer/input_layer/shopping_level_embedding/shopping_level_embedding_weights/concat/axisConst*
dtype0*
_output_shapes
: *
value	B : 
§
Xinput_layer/input_layer/shopping_level_embedding/shopping_level_embedding_weights/concatConcatV2Yinput_layer/input_layer/shopping_level_embedding/shopping_level_embedding_weights/Slice_1Yinput_layer/input_layer/shopping_level_embedding/shopping_level_embedding_weights/Slice_2]input_layer/input_layer/shopping_level_embedding/shopping_level_embedding_weights/concat/axis*
N*
T0*
_output_shapes
:
Ĺ
[input_layer/input_layer/shopping_level_embedding/shopping_level_embedding_weights/Reshape_2ReshapeQinput_layer/input_layer/shopping_level_embedding/shopping_level_embedding_weightsXinput_layer/input_layer/shopping_level_embedding/shopping_level_embedding_weights/concat*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Á
6input_layer/input_layer/shopping_level_embedding/ShapeShape[input_layer/input_layer/shopping_level_embedding/shopping_level_embedding_weights/Reshape_2*
_output_shapes
:*
T0

Dinput_layer/input_layer/shopping_level_embedding/strided_slice/stackConst*
_output_shapes
:*
valueB: *
dtype0

Finput_layer/input_layer/shopping_level_embedding/strided_slice/stack_1Const*
dtype0*
valueB:*
_output_shapes
:

Finput_layer/input_layer/shopping_level_embedding/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
˘
>input_layer/input_layer/shopping_level_embedding/strided_sliceStridedSlice6input_layer/input_layer/shopping_level_embedding/ShapeDinput_layer/input_layer/shopping_level_embedding/strided_slice/stackFinput_layer/input_layer/shopping_level_embedding/strided_slice/stack_1Finput_layer/input_layer/shopping_level_embedding/strided_slice/stack_2*
T0*
shrink_axis_mask*
_output_shapes
: *
Index0

@input_layer/input_layer/shopping_level_embedding/Reshape/shape/1Const*
_output_shapes
: *
value	B :*
dtype0
ö
>input_layer/input_layer/shopping_level_embedding/Reshape/shapePack>input_layer/input_layer/shopping_level_embedding/strided_slice@input_layer/input_layer/shopping_level_embedding/Reshape/shape/1*
_output_shapes
:*
T0*
N

8input_layer/input_layer/shopping_level_embedding/ReshapeReshape[input_layer/input_layer/shopping_level_embedding/shopping_level_embedding_weights/Reshape_2>input_layer/input_layer/shopping_level_embedding/Reshape/shape*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

8input_layer/input_layer/user_id_embedding/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙
˝
4input_layer/input_layer/user_id_embedding/ExpandDims
ExpandDimsPlaceholder_88input_layer/input_layer/user_id_embedding/ExpandDims/dim*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

Hinput_layer/input_layer/user_id_embedding/to_sparse_input/ignore_value/xConst*
valueB B *
_output_shapes
: *
dtype0

Binput_layer/input_layer/user_id_embedding/to_sparse_input/NotEqualNotEqual4input_layer/input_layer/user_id_embedding/ExpandDimsHinput_layer/input_layer/user_id_embedding/to_sparse_input/ignore_value/x*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
ˇ
Ainput_layer/input_layer/user_id_embedding/to_sparse_input/indicesWhereBinput_layer/input_layer/user_id_embedding/to_sparse_input/NotEqual*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

@input_layer/input_layer/user_id_embedding/to_sparse_input/valuesGatherNd4input_layer/input_layer/user_id_embedding/ExpandDimsAinput_layer/input_layer/user_id_embedding/to_sparse_input/indices*
Tparams0*
Tindices0	*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
š
Einput_layer/input_layer/user_id_embedding/to_sparse_input/dense_shapeShape4input_layer/input_layer/user_id_embedding/ExpandDims*
T0*
out_type0	*
_output_shapes
:
Ć
0input_layer/input_layer/user_id_embedding/lookupStringToHashBucketFast@input_layer/input_layer/user_id_embedding/to_sparse_input/values*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
num_buckets 

einput_layer/input_layer/user_id_embedding/embedding_weights/part_0/Initializer/truncated_normal/shapeConst*
valueB"     *U
_classK
IGloc:@input_layer/input_layer/user_id_embedding/embedding_weights/part_0*
dtype0*
_output_shapes
:

dinput_layer/input_layer/user_id_embedding/embedding_weights/part_0/Initializer/truncated_normal/meanConst*
valueB
 *    *
dtype0*U
_classK
IGloc:@input_layer/input_layer/user_id_embedding/embedding_weights/part_0*
_output_shapes
: 

finput_layer/input_layer/user_id_embedding/embedding_weights/part_0/Initializer/truncated_normal/stddevConst*
_output_shapes
: *
valueB
 *  >*U
_classK
IGloc:@input_layer/input_layer/user_id_embedding/embedding_weights/part_0*
dtype0
ř
oinput_layer/input_layer/user_id_embedding/embedding_weights/part_0/Initializer/truncated_normal/TruncatedNormalTruncatedNormaleinput_layer/input_layer/user_id_embedding/embedding_weights/part_0/Initializer/truncated_normal/shape*
T0* 
_output_shapes
:
 *U
_classK
IGloc:@input_layer/input_layer/user_id_embedding/embedding_weights/part_0*
dtype0
Ĺ
cinput_layer/input_layer/user_id_embedding/embedding_weights/part_0/Initializer/truncated_normal/mulMuloinput_layer/input_layer/user_id_embedding/embedding_weights/part_0/Initializer/truncated_normal/TruncatedNormalfinput_layer/input_layer/user_id_embedding/embedding_weights/part_0/Initializer/truncated_normal/stddev*U
_classK
IGloc:@input_layer/input_layer/user_id_embedding/embedding_weights/part_0* 
_output_shapes
:
 *
T0
ł
_input_layer/input_layer/user_id_embedding/embedding_weights/part_0/Initializer/truncated_normalAddcinput_layer/input_layer/user_id_embedding/embedding_weights/part_0/Initializer/truncated_normal/muldinput_layer/input_layer/user_id_embedding/embedding_weights/part_0/Initializer/truncated_normal/mean* 
_output_shapes
:
 *
T0*U
_classK
IGloc:@input_layer/input_layer/user_id_embedding/embedding_weights/part_0
í
Binput_layer/input_layer/user_id_embedding/embedding_weights/part_0
VariableV2*
shape:
 *
dtype0* 
_output_shapes
:
 *U
_classK
IGloc:@input_layer/input_layer/user_id_embedding/embedding_weights/part_0
ú
Iinput_layer/input_layer/user_id_embedding/embedding_weights/part_0/AssignAssignBinput_layer/input_layer/user_id_embedding/embedding_weights/part_0_input_layer/input_layer/user_id_embedding/embedding_weights/part_0/Initializer/truncated_normal*U
_classK
IGloc:@input_layer/input_layer/user_id_embedding/embedding_weights/part_0*
T0* 
_output_shapes
:
 

Ginput_layer/input_layer/user_id_embedding/embedding_weights/part_0/readIdentityBinput_layer/input_layer/user_id_embedding/embedding_weights/part_0*
T0*U
_classK
IGloc:@input_layer/input_layer/user_id_embedding/embedding_weights/part_0* 
_output_shapes
:
 

Oinput_layer/input_layer/user_id_embedding/user_id_embedding_weights/Slice/beginConst*
valueB: *
dtype0*
_output_shapes
:

Ninput_layer/input_layer/user_id_embedding/user_id_embedding_weights/Slice/sizeConst*
valueB:*
_output_shapes
:*
dtype0
ě
Iinput_layer/input_layer/user_id_embedding/user_id_embedding_weights/SliceSliceEinput_layer/input_layer/user_id_embedding/to_sparse_input/dense_shapeOinput_layer/input_layer/user_id_embedding/user_id_embedding_weights/Slice/beginNinput_layer/input_layer/user_id_embedding/user_id_embedding_weights/Slice/size*
T0	*
_output_shapes
:*
Index0

Iinput_layer/input_layer/user_id_embedding/user_id_embedding_weights/ConstConst*
dtype0*
valueB: *
_output_shapes
:

Hinput_layer/input_layer/user_id_embedding/user_id_embedding_weights/ProdProdIinput_layer/input_layer/user_id_embedding/user_id_embedding_weights/SliceIinput_layer/input_layer/user_id_embedding/user_id_embedding_weights/Const*
T0	*
_output_shapes
: 

Tinput_layer/input_layer/user_id_embedding/user_id_embedding_weights/GatherV2/indicesConst*
dtype0*
value	B :*
_output_shapes
: 

Qinput_layer/input_layer/user_id_embedding/user_id_embedding_weights/GatherV2/axisConst*
value	B : *
dtype0*
_output_shapes
: 

Linput_layer/input_layer/user_id_embedding/user_id_embedding_weights/GatherV2GatherV2Einput_layer/input_layer/user_id_embedding/to_sparse_input/dense_shapeTinput_layer/input_layer/user_id_embedding/user_id_embedding_weights/GatherV2/indicesQinput_layer/input_layer/user_id_embedding/user_id_embedding_weights/GatherV2/axis*
Tindices0*
Taxis0*
_output_shapes
: *
Tparams0	

Jinput_layer/input_layer/user_id_embedding/user_id_embedding_weights/Cast/xPackHinput_layer/input_layer/user_id_embedding/user_id_embedding_weights/ProdLinput_layer/input_layer/user_id_embedding/user_id_embedding_weights/GatherV2*
N*
_output_shapes
:*
T0	
ç
Qinput_layer/input_layer/user_id_embedding/user_id_embedding_weights/SparseReshapeSparseReshapeAinput_layer/input_layer/user_id_embedding/to_sparse_input/indicesEinput_layer/input_layer/user_id_embedding/to_sparse_input/dense_shapeJinput_layer/input_layer/user_id_embedding/user_id_embedding_weights/Cast/x*-
_output_shapes
:˙˙˙˙˙˙˙˙˙:
Ć
Zinput_layer/input_layer/user_id_embedding/user_id_embedding_weights/SparseReshape/IdentityIdentity0input_layer/input_layer/user_id_embedding/lookup*
T0	*#
_output_shapes
:˙˙˙˙˙˙˙˙˙

Rinput_layer/input_layer/user_id_embedding/user_id_embedding_weights/GreaterEqual/yConst*
value	B	 R *
_output_shapes
: *
dtype0	
ž
Pinput_layer/input_layer/user_id_embedding/user_id_embedding_weights/GreaterEqualGreaterEqualZinput_layer/input_layer/user_id_embedding/user_id_embedding_weights/SparseReshape/IdentityRinput_layer/input_layer/user_id_embedding/user_id_embedding_weights/GreaterEqual/y*
T0	*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
Í
Iinput_layer/input_layer/user_id_embedding/user_id_embedding_weights/WhereWherePinput_layer/input_layer/user_id_embedding/user_id_embedding_weights/GreaterEqual*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
¤
Qinput_layer/input_layer/user_id_embedding/user_id_embedding_weights/Reshape/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
_output_shapes
:*
dtype0
˘
Kinput_layer/input_layer/user_id_embedding/user_id_embedding_weights/ReshapeReshapeIinput_layer/input_layer/user_id_embedding/user_id_embedding_weights/WhereQinput_layer/input_layer/user_id_embedding/user_id_embedding_weights/Reshape/shape*
T0	*#
_output_shapes
:˙˙˙˙˙˙˙˙˙

Sinput_layer/input_layer/user_id_embedding/user_id_embedding_weights/GatherV2_1/axisConst*
dtype0*
_output_shapes
: *
value	B : 
¤
Ninput_layer/input_layer/user_id_embedding/user_id_embedding_weights/GatherV2_1GatherV2Qinput_layer/input_layer/user_id_embedding/user_id_embedding_weights/SparseReshapeKinput_layer/input_layer/user_id_embedding/user_id_embedding_weights/ReshapeSinput_layer/input_layer/user_id_embedding/user_id_embedding_weights/GatherV2_1/axis*
Tparams0	*
Tindices0	*
Taxis0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

Sinput_layer/input_layer/user_id_embedding/user_id_embedding_weights/GatherV2_2/axisConst*
_output_shapes
: *
value	B : *
dtype0
Š
Ninput_layer/input_layer/user_id_embedding/user_id_embedding_weights/GatherV2_2GatherV2Zinput_layer/input_layer/user_id_embedding/user_id_embedding_weights/SparseReshape/IdentityKinput_layer/input_layer/user_id_embedding/user_id_embedding_weights/ReshapeSinput_layer/input_layer/user_id_embedding/user_id_embedding_weights/GatherV2_2/axis*
Tparams0	*
Taxis0*
Tindices0	*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ň
Linput_layer/input_layer/user_id_embedding/user_id_embedding_weights/IdentityIdentitySinput_layer/input_layer/user_id_embedding/user_id_embedding_weights/SparseReshape:1*
T0	*
_output_shapes
:

]input_layer/input_layer/user_id_embedding/user_id_embedding_weights/SparseFillEmptyRows/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R 
Ž
kinput_layer/input_layer/user_id_embedding/user_id_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowsSparseFillEmptyRowsNinput_layer/input_layer/user_id_embedding/user_id_embedding_weights/GatherV2_1Ninput_layer/input_layer/user_id_embedding/user_id_embedding_weights/GatherV2_2Linput_layer/input_layer/user_id_embedding/user_id_embedding_weights/Identity]input_layer/input_layer/user_id_embedding/user_id_embedding_weights/SparseFillEmptyRows/Const*
T0	*T
_output_shapesB
@:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
Ŕ
oinput_layer/input_layer/user_id_embedding/user_id_embedding_weights/embedding_lookup_sparse/strided_slice/stackConst*
valueB"        *
dtype0*
_output_shapes
:
Â
qinput_layer/input_layer/user_id_embedding/user_id_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1Const*
valueB"       *
_output_shapes
:*
dtype0
Â
qinput_layer/input_layer/user_id_embedding/user_id_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB"      
˛
iinput_layer/input_layer/user_id_embedding/user_id_embedding_weights/embedding_lookup_sparse/strided_sliceStridedSlicekinput_layer/input_layer/user_id_embedding/user_id_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowsoinput_layer/input_layer/user_id_embedding/user_id_embedding_weights/embedding_lookup_sparse/strided_slice/stackqinput_layer/input_layer/user_id_embedding/user_id_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1qinput_layer/input_layer/user_id_embedding/user_id_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2*
T0	*
Index0*

begin_mask*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
end_mask*
shrink_axis_mask

`input_layer/input_layer/user_id_embedding/user_id_embedding_weights/embedding_lookup_sparse/CastCastiinput_layer/input_layer/user_id_embedding/user_id_embedding_weights/embedding_lookup_sparse/strided_slice*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*

DstT0*

SrcT0	

binput_layer/input_layer/user_id_embedding/user_id_embedding_weights/embedding_lookup_sparse/UniqueUniqueminput_layer/input_layer/user_id_embedding/user_id_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:1*
T0	*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙

qinput_layer/input_layer/user_id_embedding/user_id_embedding_weights/embedding_lookup_sparse/embedding_lookup/axisConst*
dtype0*
_output_shapes
: *U
_classK
IGloc:@input_layer/input_layer/user_id_embedding/embedding_weights/part_0*
value	B : 
Ä
linput_layer/input_layer/user_id_embedding/user_id_embedding_weights/embedding_lookup_sparse/embedding_lookupGatherV2Ginput_layer/input_layer/user_id_embedding/embedding_weights/part_0/readbinput_layer/input_layer/user_id_embedding/user_id_embedding_weights/embedding_lookup_sparse/Uniqueqinput_layer/input_layer/user_id_embedding/user_id_embedding_weights/embedding_lookup_sparse/embedding_lookup/axis*
Tparams0*
Tindices0	*U
_classK
IGloc:@input_layer/input_layer/user_id_embedding/embedding_weights/part_0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
Taxis0
Ą
uinput_layer/input_layer/user_id_embedding/user_id_embedding_weights/embedding_lookup_sparse/embedding_lookup/IdentityIdentitylinput_layer/input_layer/user_id_embedding/user_id_embedding_weights/embedding_lookup_sparse/embedding_lookup*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
á
[input_layer/input_layer/user_id_embedding/user_id_embedding_weights/embedding_lookup_sparseSparseSegmentMeanuinput_layer/input_layer/user_id_embedding/user_id_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identitydinput_layer/input_layer/user_id_embedding/user_id_embedding_weights/embedding_lookup_sparse/Unique:1`input_layer/input_layer/user_id_embedding/user_id_embedding_weights/embedding_lookup_sparse/Cast*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
¤
Sinput_layer/input_layer/user_id_embedding/user_id_embedding_weights/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   
Î
Minput_layer/input_layer/user_id_embedding/user_id_embedding_weights/Reshape_1Reshapeminput_layer/input_layer/user_id_embedding/user_id_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:2Sinput_layer/input_layer/user_id_embedding/user_id_embedding_weights/Reshape_1/shape*
T0
*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ô
Iinput_layer/input_layer/user_id_embedding/user_id_embedding_weights/ShapeShape[input_layer/input_layer/user_id_embedding/user_id_embedding_weights/embedding_lookup_sparse*
T0*
_output_shapes
:
Ą
Winput_layer/input_layer/user_id_embedding/user_id_embedding_weights/strided_slice/stackConst*
valueB:*
_output_shapes
:*
dtype0
Ł
Yinput_layer/input_layer/user_id_embedding/user_id_embedding_weights/strided_slice/stack_1Const*
valueB:*
_output_shapes
:*
dtype0
Ł
Yinput_layer/input_layer/user_id_embedding/user_id_embedding_weights/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:

Qinput_layer/input_layer/user_id_embedding/user_id_embedding_weights/strided_sliceStridedSliceIinput_layer/input_layer/user_id_embedding/user_id_embedding_weights/ShapeWinput_layer/input_layer/user_id_embedding/user_id_embedding_weights/strided_slice/stackYinput_layer/input_layer/user_id_embedding/user_id_embedding_weights/strided_slice/stack_1Yinput_layer/input_layer/user_id_embedding/user_id_embedding_weights/strided_slice/stack_2*
T0*
shrink_axis_mask*
_output_shapes
: *
Index0

Kinput_layer/input_layer/user_id_embedding/user_id_embedding_weights/stack/0Const*
value	B :*
dtype0*
_output_shapes
: 

Iinput_layer/input_layer/user_id_embedding/user_id_embedding_weights/stackPackKinput_layer/input_layer/user_id_embedding/user_id_embedding_weights/stack/0Qinput_layer/input_layer/user_id_embedding/user_id_embedding_weights/strided_slice*
N*
_output_shapes
:*
T0
Ľ
Hinput_layer/input_layer/user_id_embedding/user_id_embedding_weights/TileTileMinput_layer/input_layer/user_id_embedding/user_id_embedding_weights/Reshape_1Iinput_layer/input_layer/user_id_embedding/user_id_embedding_weights/stack*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
T0

ę
Ninput_layer/input_layer/user_id_embedding/user_id_embedding_weights/zeros_like	ZerosLike[input_layer/input_layer/user_id_embedding/user_id_embedding_weights/embedding_lookup_sparse*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
ö
Cinput_layer/input_layer/user_id_embedding/user_id_embedding_weightsSelectHinput_layer/input_layer/user_id_embedding/user_id_embedding_weights/TileNinput_layer/input_layer/user_id_embedding/user_id_embedding_weights/zeros_like[input_layer/input_layer/user_id_embedding/user_id_embedding_weights/embedding_lookup_sparse*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Í
Jinput_layer/input_layer/user_id_embedding/user_id_embedding_weights/Cast_1CastEinput_layer/input_layer/user_id_embedding/to_sparse_input/dense_shape*
_output_shapes
:*

SrcT0	*

DstT0

Qinput_layer/input_layer/user_id_embedding/user_id_embedding_weights/Slice_1/beginConst*
_output_shapes
:*
valueB: *
dtype0

Pinput_layer/input_layer/user_id_embedding/user_id_embedding_weights/Slice_1/sizeConst*
valueB:*
_output_shapes
:*
dtype0
÷
Kinput_layer/input_layer/user_id_embedding/user_id_embedding_weights/Slice_1SliceJinput_layer/input_layer/user_id_embedding/user_id_embedding_weights/Cast_1Qinput_layer/input_layer/user_id_embedding/user_id_embedding_weights/Slice_1/beginPinput_layer/input_layer/user_id_embedding/user_id_embedding_weights/Slice_1/size*
T0*
Index0*
_output_shapes
:
ž
Kinput_layer/input_layer/user_id_embedding/user_id_embedding_weights/Shape_1ShapeCinput_layer/input_layer/user_id_embedding/user_id_embedding_weights*
_output_shapes
:*
T0

Qinput_layer/input_layer/user_id_embedding/user_id_embedding_weights/Slice_2/beginConst*
dtype0*
_output_shapes
:*
valueB:
Ł
Pinput_layer/input_layer/user_id_embedding/user_id_embedding_weights/Slice_2/sizeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
_output_shapes
:*
dtype0
ř
Kinput_layer/input_layer/user_id_embedding/user_id_embedding_weights/Slice_2SliceKinput_layer/input_layer/user_id_embedding/user_id_embedding_weights/Shape_1Qinput_layer/input_layer/user_id_embedding/user_id_embedding_weights/Slice_2/beginPinput_layer/input_layer/user_id_embedding/user_id_embedding_weights/Slice_2/size*
_output_shapes
:*
Index0*
T0

Oinput_layer/input_layer/user_id_embedding/user_id_embedding_weights/concat/axisConst*
dtype0*
value	B : *
_output_shapes
: 
ď
Jinput_layer/input_layer/user_id_embedding/user_id_embedding_weights/concatConcatV2Kinput_layer/input_layer/user_id_embedding/user_id_embedding_weights/Slice_1Kinput_layer/input_layer/user_id_embedding/user_id_embedding_weights/Slice_2Oinput_layer/input_layer/user_id_embedding/user_id_embedding_weights/concat/axis*
N*
T0*
_output_shapes
:

Minput_layer/input_layer/user_id_embedding/user_id_embedding_weights/Reshape_2ReshapeCinput_layer/input_layer/user_id_embedding/user_id_embedding_weightsJinput_layer/input_layer/user_id_embedding/user_id_embedding_weights/concat*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ź
/input_layer/input_layer/user_id_embedding/ShapeShapeMinput_layer/input_layer/user_id_embedding/user_id_embedding_weights/Reshape_2*
_output_shapes
:*
T0

=input_layer/input_layer/user_id_embedding/strided_slice/stackConst*
_output_shapes
:*
valueB: *
dtype0

?input_layer/input_layer/user_id_embedding/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:

?input_layer/input_layer/user_id_embedding/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
˙
7input_layer/input_layer/user_id_embedding/strided_sliceStridedSlice/input_layer/input_layer/user_id_embedding/Shape=input_layer/input_layer/user_id_embedding/strided_slice/stack?input_layer/input_layer/user_id_embedding/strided_slice/stack_1?input_layer/input_layer/user_id_embedding/strided_slice/stack_2*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: 
{
9input_layer/input_layer/user_id_embedding/Reshape/shape/1Const*
dtype0*
_output_shapes
: *
value	B :
á
7input_layer/input_layer/user_id_embedding/Reshape/shapePack7input_layer/input_layer/user_id_embedding/strided_slice9input_layer/input_layer/user_id_embedding/Reshape/shape/1*
_output_shapes
:*
N*
T0
ö
1input_layer/input_layer/user_id_embedding/ReshapeReshapeMinput_layer/input_layer/user_id_embedding/user_id_embedding_weights/Reshape_27input_layer/input_layer/user_id_embedding/Reshape/shape*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
e
#input_layer/input_layer/concat/axisConst*
value	B :*
dtype0*
_output_shapes
: 
Č
input_layer/input_layer/concatConcatV23input_layer/input_layer/age_level_embedding/Reshape6input_layer/input_layer/cms_group_id_embedding/Reshape3input_layer/input_layer/cms_segid_embedding/Reshape>input_layer/input_layer/new_user_class_level_embedding/Reshape4input_layer/input_layer/occupation_embedding/Reshape6input_layer/input_layer/pvalue_level_embedding/Reshape8input_layer/input_layer/shopping_level_embedding/Reshape1input_layer/input_layer/user_id_embedding/Reshape#input_layer/input_layer/concat/axis*
T0*
N*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

=input_layer/input_layer_1/adgroup_id_embedding/ExpandDims/dimConst*
dtype0*
_output_shapes
: *
valueB :
˙˙˙˙˙˙˙˙˙
Ç
9input_layer/input_layer_1/adgroup_id_embedding/ExpandDims
ExpandDimsPlaceholder_3=input_layer/input_layer_1/adgroup_id_embedding/ExpandDims/dim*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

Minput_layer/input_layer_1/adgroup_id_embedding/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
valueB B *
dtype0

Ginput_layer/input_layer_1/adgroup_id_embedding/to_sparse_input/NotEqualNotEqual9input_layer/input_layer_1/adgroup_id_embedding/ExpandDimsMinput_layer/input_layer_1/adgroup_id_embedding/to_sparse_input/ignore_value/x*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Á
Finput_layer/input_layer_1/adgroup_id_embedding/to_sparse_input/indicesWhereGinput_layer/input_layer_1/adgroup_id_embedding/to_sparse_input/NotEqual*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

Einput_layer/input_layer_1/adgroup_id_embedding/to_sparse_input/valuesGatherNd9input_layer/input_layer_1/adgroup_id_embedding/ExpandDimsFinput_layer/input_layer_1/adgroup_id_embedding/to_sparse_input/indices*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
Tindices0	*
Tparams0
Ă
Jinput_layer/input_layer_1/adgroup_id_embedding/to_sparse_input/dense_shapeShape9input_layer/input_layer_1/adgroup_id_embedding/ExpandDims*
_output_shapes
:*
out_type0	*
T0
Đ
5input_layer/input_layer_1/adgroup_id_embedding/lookupStringToHashBucketFastEinput_layer/input_layer_1/adgroup_id_embedding/to_sparse_input/values*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
num_buckets 

jinput_layer/input_layer_1/adgroup_id_embedding/embedding_weights/part_0/Initializer/truncated_normal/shapeConst*Z
_classP
NLloc:@input_layer/input_layer_1/adgroup_id_embedding/embedding_weights/part_0*
dtype0*
valueB"     *
_output_shapes
:

iinput_layer/input_layer_1/adgroup_id_embedding/embedding_weights/part_0/Initializer/truncated_normal/meanConst*
dtype0*
_output_shapes
: *
valueB
 *    *Z
_classP
NLloc:@input_layer/input_layer_1/adgroup_id_embedding/embedding_weights/part_0

kinput_layer/input_layer_1/adgroup_id_embedding/embedding_weights/part_0/Initializer/truncated_normal/stddevConst*Z
_classP
NLloc:@input_layer/input_layer_1/adgroup_id_embedding/embedding_weights/part_0*
dtype0*
_output_shapes
: *
valueB
 *  >

tinput_layer/input_layer_1/adgroup_id_embedding/embedding_weights/part_0/Initializer/truncated_normal/TruncatedNormalTruncatedNormaljinput_layer/input_layer_1/adgroup_id_embedding/embedding_weights/part_0/Initializer/truncated_normal/shape*Z
_classP
NLloc:@input_layer/input_layer_1/adgroup_id_embedding/embedding_weights/part_0*
T0* 
_output_shapes
:
 *
dtype0
Ů
hinput_layer/input_layer_1/adgroup_id_embedding/embedding_weights/part_0/Initializer/truncated_normal/mulMultinput_layer/input_layer_1/adgroup_id_embedding/embedding_weights/part_0/Initializer/truncated_normal/TruncatedNormalkinput_layer/input_layer_1/adgroup_id_embedding/embedding_weights/part_0/Initializer/truncated_normal/stddev*Z
_classP
NLloc:@input_layer/input_layer_1/adgroup_id_embedding/embedding_weights/part_0*
T0* 
_output_shapes
:
 
Ç
dinput_layer/input_layer_1/adgroup_id_embedding/embedding_weights/part_0/Initializer/truncated_normalAddhinput_layer/input_layer_1/adgroup_id_embedding/embedding_weights/part_0/Initializer/truncated_normal/muliinput_layer/input_layer_1/adgroup_id_embedding/embedding_weights/part_0/Initializer/truncated_normal/mean* 
_output_shapes
:
 *Z
_classP
NLloc:@input_layer/input_layer_1/adgroup_id_embedding/embedding_weights/part_0*
T0
÷
Ginput_layer/input_layer_1/adgroup_id_embedding/embedding_weights/part_0
VariableV2*Z
_classP
NLloc:@input_layer/input_layer_1/adgroup_id_embedding/embedding_weights/part_0* 
_output_shapes
:
 *
shape:
 *
dtype0

Ninput_layer/input_layer_1/adgroup_id_embedding/embedding_weights/part_0/AssignAssignGinput_layer/input_layer_1/adgroup_id_embedding/embedding_weights/part_0dinput_layer/input_layer_1/adgroup_id_embedding/embedding_weights/part_0/Initializer/truncated_normal* 
_output_shapes
:
 *
T0*Z
_classP
NLloc:@input_layer/input_layer_1/adgroup_id_embedding/embedding_weights/part_0
¨
Linput_layer/input_layer_1/adgroup_id_embedding/embedding_weights/part_0/readIdentityGinput_layer/input_layer_1/adgroup_id_embedding/embedding_weights/part_0*Z
_classP
NLloc:@input_layer/input_layer_1/adgroup_id_embedding/embedding_weights/part_0*
T0* 
_output_shapes
:
 
Ą
Winput_layer/input_layer_1/adgroup_id_embedding/adgroup_id_embedding_weights/Slice/beginConst*
valueB: *
dtype0*
_output_shapes
:
 
Vinput_layer/input_layer_1/adgroup_id_embedding/adgroup_id_embedding_weights/Slice/sizeConst*
valueB:*
dtype0*
_output_shapes
:

Qinput_layer/input_layer_1/adgroup_id_embedding/adgroup_id_embedding_weights/SliceSliceJinput_layer/input_layer_1/adgroup_id_embedding/to_sparse_input/dense_shapeWinput_layer/input_layer_1/adgroup_id_embedding/adgroup_id_embedding_weights/Slice/beginVinput_layer/input_layer_1/adgroup_id_embedding/adgroup_id_embedding_weights/Slice/size*
T0	*
_output_shapes
:*
Index0

Qinput_layer/input_layer_1/adgroup_id_embedding/adgroup_id_embedding_weights/ConstConst*
valueB: *
dtype0*
_output_shapes
:

Pinput_layer/input_layer_1/adgroup_id_embedding/adgroup_id_embedding_weights/ProdProdQinput_layer/input_layer_1/adgroup_id_embedding/adgroup_id_embedding_weights/SliceQinput_layer/input_layer_1/adgroup_id_embedding/adgroup_id_embedding_weights/Const*
_output_shapes
: *
T0	

\input_layer/input_layer_1/adgroup_id_embedding/adgroup_id_embedding_weights/GatherV2/indicesConst*
dtype0*
value	B :*
_output_shapes
: 

Yinput_layer/input_layer_1/adgroup_id_embedding/adgroup_id_embedding_weights/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 
Š
Tinput_layer/input_layer_1/adgroup_id_embedding/adgroup_id_embedding_weights/GatherV2GatherV2Jinput_layer/input_layer_1/adgroup_id_embedding/to_sparse_input/dense_shape\input_layer/input_layer_1/adgroup_id_embedding/adgroup_id_embedding_weights/GatherV2/indicesYinput_layer/input_layer_1/adgroup_id_embedding/adgroup_id_embedding_weights/GatherV2/axis*
_output_shapes
: *
Taxis0*
Tparams0	*
Tindices0
°
Rinput_layer/input_layer_1/adgroup_id_embedding/adgroup_id_embedding_weights/Cast/xPackPinput_layer/input_layer_1/adgroup_id_embedding/adgroup_id_embedding_weights/ProdTinput_layer/input_layer_1/adgroup_id_embedding/adgroup_id_embedding_weights/GatherV2*
_output_shapes
:*
T0	*
N

Yinput_layer/input_layer_1/adgroup_id_embedding/adgroup_id_embedding_weights/SparseReshapeSparseReshapeFinput_layer/input_layer_1/adgroup_id_embedding/to_sparse_input/indicesJinput_layer/input_layer_1/adgroup_id_embedding/to_sparse_input/dense_shapeRinput_layer/input_layer_1/adgroup_id_embedding/adgroup_id_embedding_weights/Cast/x*-
_output_shapes
:˙˙˙˙˙˙˙˙˙:
Ó
binput_layer/input_layer_1/adgroup_id_embedding/adgroup_id_embedding_weights/SparseReshape/IdentityIdentity5input_layer/input_layer_1/adgroup_id_embedding/lookup*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0	

Zinput_layer/input_layer_1/adgroup_id_embedding/adgroup_id_embedding_weights/GreaterEqual/yConst*
_output_shapes
: *
value	B	 R *
dtype0	
Ö
Xinput_layer/input_layer_1/adgroup_id_embedding/adgroup_id_embedding_weights/GreaterEqualGreaterEqualbinput_layer/input_layer_1/adgroup_id_embedding/adgroup_id_embedding_weights/SparseReshape/IdentityZinput_layer/input_layer_1/adgroup_id_embedding/adgroup_id_embedding_weights/GreaterEqual/y*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0	
Ý
Qinput_layer/input_layer_1/adgroup_id_embedding/adgroup_id_embedding_weights/WhereWhereXinput_layer/input_layer_1/adgroup_id_embedding/adgroup_id_embedding_weights/GreaterEqual*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ź
Yinput_layer/input_layer_1/adgroup_id_embedding/adgroup_id_embedding_weights/Reshape/shapeConst*
dtype0*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙
ş
Sinput_layer/input_layer_1/adgroup_id_embedding/adgroup_id_embedding_weights/ReshapeReshapeQinput_layer/input_layer_1/adgroup_id_embedding/adgroup_id_embedding_weights/WhereYinput_layer/input_layer_1/adgroup_id_embedding/adgroup_id_embedding_weights/Reshape/shape*
T0	*#
_output_shapes
:˙˙˙˙˙˙˙˙˙

[input_layer/input_layer_1/adgroup_id_embedding/adgroup_id_embedding_weights/GatherV2_1/axisConst*
dtype0*
_output_shapes
: *
value	B : 
Ä
Vinput_layer/input_layer_1/adgroup_id_embedding/adgroup_id_embedding_weights/GatherV2_1GatherV2Yinput_layer/input_layer_1/adgroup_id_embedding/adgroup_id_embedding_weights/SparseReshapeSinput_layer/input_layer_1/adgroup_id_embedding/adgroup_id_embedding_weights/Reshape[input_layer/input_layer_1/adgroup_id_embedding/adgroup_id_embedding_weights/GatherV2_1/axis*
Taxis0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
Tindices0	*
Tparams0	

[input_layer/input_layer_1/adgroup_id_embedding/adgroup_id_embedding_weights/GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
value	B : 
É
Vinput_layer/input_layer_1/adgroup_id_embedding/adgroup_id_embedding_weights/GatherV2_2GatherV2binput_layer/input_layer_1/adgroup_id_embedding/adgroup_id_embedding_weights/SparseReshape/IdentitySinput_layer/input_layer_1/adgroup_id_embedding/adgroup_id_embedding_weights/Reshape[input_layer/input_layer_1/adgroup_id_embedding/adgroup_id_embedding_weights/GatherV2_2/axis*
Tparams0	*
Tindices0	*
Taxis0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
â
Tinput_layer/input_layer_1/adgroup_id_embedding/adgroup_id_embedding_weights/IdentityIdentity[input_layer/input_layer_1/adgroup_id_embedding/adgroup_id_embedding_weights/SparseReshape:1*
_output_shapes
:*
T0	
§
einput_layer/input_layer_1/adgroup_id_embedding/adgroup_id_embedding_weights/SparseFillEmptyRows/ConstConst*
value	B	 R *
_output_shapes
: *
dtype0	
Ö
sinput_layer/input_layer_1/adgroup_id_embedding/adgroup_id_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowsSparseFillEmptyRowsVinput_layer/input_layer_1/adgroup_id_embedding/adgroup_id_embedding_weights/GatherV2_1Vinput_layer/input_layer_1/adgroup_id_embedding/adgroup_id_embedding_weights/GatherV2_2Tinput_layer/input_layer_1/adgroup_id_embedding/adgroup_id_embedding_weights/Identityeinput_layer/input_layer_1/adgroup_id_embedding/adgroup_id_embedding_weights/SparseFillEmptyRows/Const*T
_output_shapesB
@:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0	
Č
winput_layer/input_layer_1/adgroup_id_embedding/adgroup_id_embedding_weights/embedding_lookup_sparse/strided_slice/stackConst*
valueB"        *
dtype0*
_output_shapes
:
Ę
yinput_layer/input_layer_1/adgroup_id_embedding/adgroup_id_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB"       
Ę
yinput_layer/input_layer_1/adgroup_id_embedding/adgroup_id_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
Ú
qinput_layer/input_layer_1/adgroup_id_embedding/adgroup_id_embedding_weights/embedding_lookup_sparse/strided_sliceStridedSlicesinput_layer/input_layer_1/adgroup_id_embedding/adgroup_id_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowswinput_layer/input_layer_1/adgroup_id_embedding/adgroup_id_embedding_weights/embedding_lookup_sparse/strided_slice/stackyinput_layer/input_layer_1/adgroup_id_embedding/adgroup_id_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1yinput_layer/input_layer_1/adgroup_id_embedding/adgroup_id_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2*
Index0*

begin_mask*
end_mask*
T0	*
shrink_axis_mask*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
hinput_layer/input_layer_1/adgroup_id_embedding/adgroup_id_embedding_weights/embedding_lookup_sparse/CastCastqinput_layer/input_layer_1/adgroup_id_embedding/adgroup_id_embedding_weights/embedding_lookup_sparse/strided_slice*

DstT0*

SrcT0	*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
¨
jinput_layer/input_layer_1/adgroup_id_embedding/adgroup_id_embedding_weights/embedding_lookup_sparse/UniqueUniqueuinput_layer/input_layer_1/adgroup_id_embedding/adgroup_id_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0	

yinput_layer/input_layer_1/adgroup_id_embedding/adgroup_id_embedding_weights/embedding_lookup_sparse/embedding_lookup/axisConst*
_output_shapes
: *
value	B : *Z
_classP
NLloc:@input_layer/input_layer_1/adgroup_id_embedding/embedding_weights/part_0*
dtype0
ć
tinput_layer/input_layer_1/adgroup_id_embedding/adgroup_id_embedding_weights/embedding_lookup_sparse/embedding_lookupGatherV2Linput_layer/input_layer_1/adgroup_id_embedding/embedding_weights/part_0/readjinput_layer/input_layer_1/adgroup_id_embedding/adgroup_id_embedding_weights/embedding_lookup_sparse/Uniqueyinput_layer/input_layer_1/adgroup_id_embedding/adgroup_id_embedding_weights/embedding_lookup_sparse/embedding_lookup/axis*Z
_classP
NLloc:@input_layer/input_layer_1/adgroup_id_embedding/embedding_weights/part_0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
Tparams0*
Taxis0*
Tindices0	
ą
}input_layer/input_layer_1/adgroup_id_embedding/adgroup_id_embedding_weights/embedding_lookup_sparse/embedding_lookup/IdentityIdentitytinput_layer/input_layer_1/adgroup_id_embedding/adgroup_id_embedding_weights/embedding_lookup_sparse/embedding_lookup*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

cinput_layer/input_layer_1/adgroup_id_embedding/adgroup_id_embedding_weights/embedding_lookup_sparseSparseSegmentMean}input_layer/input_layer_1/adgroup_id_embedding/adgroup_id_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identitylinput_layer/input_layer_1/adgroup_id_embedding/adgroup_id_embedding_weights/embedding_lookup_sparse/Unique:1hinput_layer/input_layer_1/adgroup_id_embedding/adgroup_id_embedding_weights/embedding_lookup_sparse/Cast*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ź
[input_layer/input_layer_1/adgroup_id_embedding/adgroup_id_embedding_weights/Reshape_1/shapeConst*
valueB"˙˙˙˙   *
dtype0*
_output_shapes
:
ć
Uinput_layer/input_layer_1/adgroup_id_embedding/adgroup_id_embedding_weights/Reshape_1Reshapeuinput_layer/input_layer_1/adgroup_id_embedding/adgroup_id_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:2[input_layer/input_layer_1/adgroup_id_embedding/adgroup_id_embedding_weights/Reshape_1/shape*
T0
*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
ä
Qinput_layer/input_layer_1/adgroup_id_embedding/adgroup_id_embedding_weights/ShapeShapecinput_layer/input_layer_1/adgroup_id_embedding/adgroup_id_embedding_weights/embedding_lookup_sparse*
T0*
_output_shapes
:
Š
_input_layer/input_layer_1/adgroup_id_embedding/adgroup_id_embedding_weights/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB:
Ť
ainput_layer/input_layer_1/adgroup_id_embedding/adgroup_id_embedding_weights/strided_slice/stack_1Const*
dtype0*
valueB:*
_output_shapes
:
Ť
ainput_layer/input_layer_1/adgroup_id_embedding/adgroup_id_embedding_weights/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Š
Yinput_layer/input_layer_1/adgroup_id_embedding/adgroup_id_embedding_weights/strided_sliceStridedSliceQinput_layer/input_layer_1/adgroup_id_embedding/adgroup_id_embedding_weights/Shape_input_layer/input_layer_1/adgroup_id_embedding/adgroup_id_embedding_weights/strided_slice/stackainput_layer/input_layer_1/adgroup_id_embedding/adgroup_id_embedding_weights/strided_slice/stack_1ainput_layer/input_layer_1/adgroup_id_embedding/adgroup_id_embedding_weights/strided_slice/stack_2*
T0*
_output_shapes
: *
shrink_axis_mask*
Index0

Sinput_layer/input_layer_1/adgroup_id_embedding/adgroup_id_embedding_weights/stack/0Const*
_output_shapes
: *
dtype0*
value	B :
ˇ
Qinput_layer/input_layer_1/adgroup_id_embedding/adgroup_id_embedding_weights/stackPackSinput_layer/input_layer_1/adgroup_id_embedding/adgroup_id_embedding_weights/stack/0Yinput_layer/input_layer_1/adgroup_id_embedding/adgroup_id_embedding_weights/strided_slice*
_output_shapes
:*
T0*
N
˝
Pinput_layer/input_layer_1/adgroup_id_embedding/adgroup_id_embedding_weights/TileTileUinput_layer/input_layer_1/adgroup_id_embedding/adgroup_id_embedding_weights/Reshape_1Qinput_layer/input_layer_1/adgroup_id_embedding/adgroup_id_embedding_weights/stack*
T0
*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
ú
Vinput_layer/input_layer_1/adgroup_id_embedding/adgroup_id_embedding_weights/zeros_like	ZerosLikecinput_layer/input_layer_1/adgroup_id_embedding/adgroup_id_embedding_weights/embedding_lookup_sparse*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

Kinput_layer/input_layer_1/adgroup_id_embedding/adgroup_id_embedding_weightsSelectPinput_layer/input_layer_1/adgroup_id_embedding/adgroup_id_embedding_weights/TileVinput_layer/input_layer_1/adgroup_id_embedding/adgroup_id_embedding_weights/zeros_likecinput_layer/input_layer_1/adgroup_id_embedding/adgroup_id_embedding_weights/embedding_lookup_sparse*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ú
Rinput_layer/input_layer_1/adgroup_id_embedding/adgroup_id_embedding_weights/Cast_1CastJinput_layer/input_layer_1/adgroup_id_embedding/to_sparse_input/dense_shape*

SrcT0	*
_output_shapes
:*

DstT0
Ł
Yinput_layer/input_layer_1/adgroup_id_embedding/adgroup_id_embedding_weights/Slice_1/beginConst*
_output_shapes
:*
valueB: *
dtype0
˘
Xinput_layer/input_layer_1/adgroup_id_embedding/adgroup_id_embedding_weights/Slice_1/sizeConst*
dtype0*
valueB:*
_output_shapes
:

Sinput_layer/input_layer_1/adgroup_id_embedding/adgroup_id_embedding_weights/Slice_1SliceRinput_layer/input_layer_1/adgroup_id_embedding/adgroup_id_embedding_weights/Cast_1Yinput_layer/input_layer_1/adgroup_id_embedding/adgroup_id_embedding_weights/Slice_1/beginXinput_layer/input_layer_1/adgroup_id_embedding/adgroup_id_embedding_weights/Slice_1/size*
T0*
_output_shapes
:*
Index0
Î
Sinput_layer/input_layer_1/adgroup_id_embedding/adgroup_id_embedding_weights/Shape_1ShapeKinput_layer/input_layer_1/adgroup_id_embedding/adgroup_id_embedding_weights*
_output_shapes
:*
T0
Ł
Yinput_layer/input_layer_1/adgroup_id_embedding/adgroup_id_embedding_weights/Slice_2/beginConst*
_output_shapes
:*
dtype0*
valueB:
Ť
Xinput_layer/input_layer_1/adgroup_id_embedding/adgroup_id_embedding_weights/Slice_2/sizeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:

Sinput_layer/input_layer_1/adgroup_id_embedding/adgroup_id_embedding_weights/Slice_2SliceSinput_layer/input_layer_1/adgroup_id_embedding/adgroup_id_embedding_weights/Shape_1Yinput_layer/input_layer_1/adgroup_id_embedding/adgroup_id_embedding_weights/Slice_2/beginXinput_layer/input_layer_1/adgroup_id_embedding/adgroup_id_embedding_weights/Slice_2/size*
T0*
Index0*
_output_shapes
:

Winput_layer/input_layer_1/adgroup_id_embedding/adgroup_id_embedding_weights/concat/axisConst*
_output_shapes
: *
value	B : *
dtype0

Rinput_layer/input_layer_1/adgroup_id_embedding/adgroup_id_embedding_weights/concatConcatV2Sinput_layer/input_layer_1/adgroup_id_embedding/adgroup_id_embedding_weights/Slice_1Sinput_layer/input_layer_1/adgroup_id_embedding/adgroup_id_embedding_weights/Slice_2Winput_layer/input_layer_1/adgroup_id_embedding/adgroup_id_embedding_weights/concat/axis*
N*
_output_shapes
:*
T0
ł
Uinput_layer/input_layer_1/adgroup_id_embedding/adgroup_id_embedding_weights/Reshape_2ReshapeKinput_layer/input_layer_1/adgroup_id_embedding/adgroup_id_embedding_weightsRinput_layer/input_layer_1/adgroup_id_embedding/adgroup_id_embedding_weights/concat*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
š
4input_layer/input_layer_1/adgroup_id_embedding/ShapeShapeUinput_layer/input_layer_1/adgroup_id_embedding/adgroup_id_embedding_weights/Reshape_2*
T0*
_output_shapes
:

Binput_layer/input_layer_1/adgroup_id_embedding/strided_slice/stackConst*
valueB: *
_output_shapes
:*
dtype0

Dinput_layer/input_layer_1/adgroup_id_embedding/strided_slice/stack_1Const*
valueB:*
_output_shapes
:*
dtype0

Dinput_layer/input_layer_1/adgroup_id_embedding/strided_slice/stack_2Const*
_output_shapes
:*
valueB:*
dtype0

<input_layer/input_layer_1/adgroup_id_embedding/strided_sliceStridedSlice4input_layer/input_layer_1/adgroup_id_embedding/ShapeBinput_layer/input_layer_1/adgroup_id_embedding/strided_slice/stackDinput_layer/input_layer_1/adgroup_id_embedding/strided_slice/stack_1Dinput_layer/input_layer_1/adgroup_id_embedding/strided_slice/stack_2*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: 

>input_layer/input_layer_1/adgroup_id_embedding/Reshape/shape/1Const*
_output_shapes
: *
value	B :*
dtype0
đ
<input_layer/input_layer_1/adgroup_id_embedding/Reshape/shapePack<input_layer/input_layer_1/adgroup_id_embedding/strided_slice>input_layer/input_layer_1/adgroup_id_embedding/Reshape/shape/1*
N*
_output_shapes
:*
T0

6input_layer/input_layer_1/adgroup_id_embedding/ReshapeReshapeUinput_layer/input_layer_1/adgroup_id_embedding/adgroup_id_embedding_weights/Reshape_2<input_layer/input_layer_1/adgroup_id_embedding/Reshape/shape*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

8input_layer/input_layer_1/brand_embedding/ExpandDims/dimConst*
dtype0*
_output_shapes
: *
valueB :
˙˙˙˙˙˙˙˙˙
˝
4input_layer/input_layer_1/brand_embedding/ExpandDims
ExpandDimsPlaceholder_78input_layer/input_layer_1/brand_embedding/ExpandDims/dim*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

Hinput_layer/input_layer_1/brand_embedding/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
valueB B *
dtype0

Binput_layer/input_layer_1/brand_embedding/to_sparse_input/NotEqualNotEqual4input_layer/input_layer_1/brand_embedding/ExpandDimsHinput_layer/input_layer_1/brand_embedding/to_sparse_input/ignore_value/x*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
ˇ
Ainput_layer/input_layer_1/brand_embedding/to_sparse_input/indicesWhereBinput_layer/input_layer_1/brand_embedding/to_sparse_input/NotEqual*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

@input_layer/input_layer_1/brand_embedding/to_sparse_input/valuesGatherNd4input_layer/input_layer_1/brand_embedding/ExpandDimsAinput_layer/input_layer_1/brand_embedding/to_sparse_input/indices*
Tindices0	*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
Tparams0
š
Einput_layer/input_layer_1/brand_embedding/to_sparse_input/dense_shapeShape4input_layer/input_layer_1/brand_embedding/ExpandDims*
_output_shapes
:*
out_type0	*
T0
Ć
0input_layer/input_layer_1/brand_embedding/lookupStringToHashBucketFast@input_layer/input_layer_1/brand_embedding/to_sparse_input/values*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
num_buckets 

einput_layer/input_layer_1/brand_embedding/embedding_weights/part_0/Initializer/truncated_normal/shapeConst*
valueB"     *
_output_shapes
:*
dtype0*U
_classK
IGloc:@input_layer/input_layer_1/brand_embedding/embedding_weights/part_0

dinput_layer/input_layer_1/brand_embedding/embedding_weights/part_0/Initializer/truncated_normal/meanConst*
dtype0*U
_classK
IGloc:@input_layer/input_layer_1/brand_embedding/embedding_weights/part_0*
_output_shapes
: *
valueB
 *    

finput_layer/input_layer_1/brand_embedding/embedding_weights/part_0/Initializer/truncated_normal/stddevConst*
dtype0*
_output_shapes
: *
valueB
 *  >*U
_classK
IGloc:@input_layer/input_layer_1/brand_embedding/embedding_weights/part_0
ř
oinput_layer/input_layer_1/brand_embedding/embedding_weights/part_0/Initializer/truncated_normal/TruncatedNormalTruncatedNormaleinput_layer/input_layer_1/brand_embedding/embedding_weights/part_0/Initializer/truncated_normal/shape*U
_classK
IGloc:@input_layer/input_layer_1/brand_embedding/embedding_weights/part_0* 
_output_shapes
:
 *
dtype0*
T0
Ĺ
cinput_layer/input_layer_1/brand_embedding/embedding_weights/part_0/Initializer/truncated_normal/mulMuloinput_layer/input_layer_1/brand_embedding/embedding_weights/part_0/Initializer/truncated_normal/TruncatedNormalfinput_layer/input_layer_1/brand_embedding/embedding_weights/part_0/Initializer/truncated_normal/stddev* 
_output_shapes
:
 *
T0*U
_classK
IGloc:@input_layer/input_layer_1/brand_embedding/embedding_weights/part_0
ł
_input_layer/input_layer_1/brand_embedding/embedding_weights/part_0/Initializer/truncated_normalAddcinput_layer/input_layer_1/brand_embedding/embedding_weights/part_0/Initializer/truncated_normal/muldinput_layer/input_layer_1/brand_embedding/embedding_weights/part_0/Initializer/truncated_normal/mean*
T0* 
_output_shapes
:
 *U
_classK
IGloc:@input_layer/input_layer_1/brand_embedding/embedding_weights/part_0
í
Binput_layer/input_layer_1/brand_embedding/embedding_weights/part_0
VariableV2* 
_output_shapes
:
 *U
_classK
IGloc:@input_layer/input_layer_1/brand_embedding/embedding_weights/part_0*
dtype0*
shape:
 
ú
Iinput_layer/input_layer_1/brand_embedding/embedding_weights/part_0/AssignAssignBinput_layer/input_layer_1/brand_embedding/embedding_weights/part_0_input_layer/input_layer_1/brand_embedding/embedding_weights/part_0/Initializer/truncated_normal*U
_classK
IGloc:@input_layer/input_layer_1/brand_embedding/embedding_weights/part_0* 
_output_shapes
:
 *
T0

Ginput_layer/input_layer_1/brand_embedding/embedding_weights/part_0/readIdentityBinput_layer/input_layer_1/brand_embedding/embedding_weights/part_0*
T0*U
_classK
IGloc:@input_layer/input_layer_1/brand_embedding/embedding_weights/part_0* 
_output_shapes
:
 

Minput_layer/input_layer_1/brand_embedding/brand_embedding_weights/Slice/beginConst*
_output_shapes
:*
dtype0*
valueB: 

Linput_layer/input_layer_1/brand_embedding/brand_embedding_weights/Slice/sizeConst*
_output_shapes
:*
valueB:*
dtype0
ć
Ginput_layer/input_layer_1/brand_embedding/brand_embedding_weights/SliceSliceEinput_layer/input_layer_1/brand_embedding/to_sparse_input/dense_shapeMinput_layer/input_layer_1/brand_embedding/brand_embedding_weights/Slice/beginLinput_layer/input_layer_1/brand_embedding/brand_embedding_weights/Slice/size*
_output_shapes
:*
T0	*
Index0

Ginput_layer/input_layer_1/brand_embedding/brand_embedding_weights/ConstConst*
_output_shapes
:*
dtype0*
valueB: 

Finput_layer/input_layer_1/brand_embedding/brand_embedding_weights/ProdProdGinput_layer/input_layer_1/brand_embedding/brand_embedding_weights/SliceGinput_layer/input_layer_1/brand_embedding/brand_embedding_weights/Const*
T0	*
_output_shapes
: 

Rinput_layer/input_layer_1/brand_embedding/brand_embedding_weights/GatherV2/indicesConst*
value	B :*
_output_shapes
: *
dtype0

Oinput_layer/input_layer_1/brand_embedding/brand_embedding_weights/GatherV2/axisConst*
dtype0*
_output_shapes
: *
value	B : 

Jinput_layer/input_layer_1/brand_embedding/brand_embedding_weights/GatherV2GatherV2Einput_layer/input_layer_1/brand_embedding/to_sparse_input/dense_shapeRinput_layer/input_layer_1/brand_embedding/brand_embedding_weights/GatherV2/indicesOinput_layer/input_layer_1/brand_embedding/brand_embedding_weights/GatherV2/axis*
Tindices0*
Tparams0	*
Taxis0*
_output_shapes
: 

Hinput_layer/input_layer_1/brand_embedding/brand_embedding_weights/Cast/xPackFinput_layer/input_layer_1/brand_embedding/brand_embedding_weights/ProdJinput_layer/input_layer_1/brand_embedding/brand_embedding_weights/GatherV2*
_output_shapes
:*
N*
T0	
ă
Oinput_layer/input_layer_1/brand_embedding/brand_embedding_weights/SparseReshapeSparseReshapeAinput_layer/input_layer_1/brand_embedding/to_sparse_input/indicesEinput_layer/input_layer_1/brand_embedding/to_sparse_input/dense_shapeHinput_layer/input_layer_1/brand_embedding/brand_embedding_weights/Cast/x*-
_output_shapes
:˙˙˙˙˙˙˙˙˙:
Ä
Xinput_layer/input_layer_1/brand_embedding/brand_embedding_weights/SparseReshape/IdentityIdentity0input_layer/input_layer_1/brand_embedding/lookup*
T0	*#
_output_shapes
:˙˙˙˙˙˙˙˙˙

Pinput_layer/input_layer_1/brand_embedding/brand_embedding_weights/GreaterEqual/yConst*
dtype0	*
_output_shapes
: *
value	B	 R 
¸
Ninput_layer/input_layer_1/brand_embedding/brand_embedding_weights/GreaterEqualGreaterEqualXinput_layer/input_layer_1/brand_embedding/brand_embedding_weights/SparseReshape/IdentityPinput_layer/input_layer_1/brand_embedding/brand_embedding_weights/GreaterEqual/y*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0	
É
Ginput_layer/input_layer_1/brand_embedding/brand_embedding_weights/WhereWhereNinput_layer/input_layer_1/brand_embedding/brand_embedding_weights/GreaterEqual*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
˘
Oinput_layer/input_layer_1/brand_embedding/brand_embedding_weights/Reshape/shapeConst*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙*
_output_shapes
:

Iinput_layer/input_layer_1/brand_embedding/brand_embedding_weights/ReshapeReshapeGinput_layer/input_layer_1/brand_embedding/brand_embedding_weights/WhereOinput_layer/input_layer_1/brand_embedding/brand_embedding_weights/Reshape/shape*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0	

Qinput_layer/input_layer_1/brand_embedding/brand_embedding_weights/GatherV2_1/axisConst*
dtype0*
value	B : *
_output_shapes
: 

Linput_layer/input_layer_1/brand_embedding/brand_embedding_weights/GatherV2_1GatherV2Oinput_layer/input_layer_1/brand_embedding/brand_embedding_weights/SparseReshapeIinput_layer/input_layer_1/brand_embedding/brand_embedding_weights/ReshapeQinput_layer/input_layer_1/brand_embedding/brand_embedding_weights/GatherV2_1/axis*
Tparams0	*
Tindices0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
Taxis0

Qinput_layer/input_layer_1/brand_embedding/brand_embedding_weights/GatherV2_2/axisConst*
value	B : *
_output_shapes
: *
dtype0
Ą
Linput_layer/input_layer_1/brand_embedding/brand_embedding_weights/GatherV2_2GatherV2Xinput_layer/input_layer_1/brand_embedding/brand_embedding_weights/SparseReshape/IdentityIinput_layer/input_layer_1/brand_embedding/brand_embedding_weights/ReshapeQinput_layer/input_layer_1/brand_embedding/brand_embedding_weights/GatherV2_2/axis*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
Tparams0	*
Tindices0	*
Taxis0
Î
Jinput_layer/input_layer_1/brand_embedding/brand_embedding_weights/IdentityIdentityQinput_layer/input_layer_1/brand_embedding/brand_embedding_weights/SparseReshape:1*
_output_shapes
:*
T0	

[input_layer/input_layer_1/brand_embedding/brand_embedding_weights/SparseFillEmptyRows/ConstConst*
value	B	 R *
dtype0	*
_output_shapes
: 
¤
iinput_layer/input_layer_1/brand_embedding/brand_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowsSparseFillEmptyRowsLinput_layer/input_layer_1/brand_embedding/brand_embedding_weights/GatherV2_1Linput_layer/input_layer_1/brand_embedding/brand_embedding_weights/GatherV2_2Jinput_layer/input_layer_1/brand_embedding/brand_embedding_weights/Identity[input_layer/input_layer_1/brand_embedding/brand_embedding_weights/SparseFillEmptyRows/Const*T
_output_shapesB
@:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0	
ž
minput_layer/input_layer_1/brand_embedding/brand_embedding_weights/embedding_lookup_sparse/strided_slice/stackConst*
_output_shapes
:*
valueB"        *
dtype0
Ŕ
oinput_layer/input_layer_1/brand_embedding/brand_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1Const*
dtype0*
valueB"       *
_output_shapes
:
Ŕ
oinput_layer/input_layer_1/brand_embedding/brand_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2Const*
_output_shapes
:*
valueB"      *
dtype0
¨
ginput_layer/input_layer_1/brand_embedding/brand_embedding_weights/embedding_lookup_sparse/strided_sliceStridedSliceiinput_layer/input_layer_1/brand_embedding/brand_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowsminput_layer/input_layer_1/brand_embedding/brand_embedding_weights/embedding_lookup_sparse/strided_slice/stackoinput_layer/input_layer_1/brand_embedding/brand_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1oinput_layer/input_layer_1/brand_embedding/brand_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2*
T0	*
shrink_axis_mask*
end_mask*
Index0*

begin_mask*#
_output_shapes
:˙˙˙˙˙˙˙˙˙

^input_layer/input_layer_1/brand_embedding/brand_embedding_weights/embedding_lookup_sparse/CastCastginput_layer/input_layer_1/brand_embedding/brand_embedding_weights/embedding_lookup_sparse/strided_slice*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*

SrcT0	*

DstT0

`input_layer/input_layer_1/brand_embedding/brand_embedding_weights/embedding_lookup_sparse/UniqueUniquekinput_layer/input_layer_1/brand_embedding/brand_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:1*
T0	*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙

oinput_layer/input_layer_1/brand_embedding/brand_embedding_weights/embedding_lookup_sparse/embedding_lookup/axisConst*
_output_shapes
: *U
_classK
IGloc:@input_layer/input_layer_1/brand_embedding/embedding_weights/part_0*
dtype0*
value	B : 
ž
jinput_layer/input_layer_1/brand_embedding/brand_embedding_weights/embedding_lookup_sparse/embedding_lookupGatherV2Ginput_layer/input_layer_1/brand_embedding/embedding_weights/part_0/read`input_layer/input_layer_1/brand_embedding/brand_embedding_weights/embedding_lookup_sparse/Uniqueoinput_layer/input_layer_1/brand_embedding/brand_embedding_weights/embedding_lookup_sparse/embedding_lookup/axis*U
_classK
IGloc:@input_layer/input_layer_1/brand_embedding/embedding_weights/part_0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
Tindices0	*
Taxis0*
Tparams0

sinput_layer/input_layer_1/brand_embedding/brand_embedding_weights/embedding_lookup_sparse/embedding_lookup/IdentityIdentityjinput_layer/input_layer_1/brand_embedding/brand_embedding_weights/embedding_lookup_sparse/embedding_lookup*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ů
Yinput_layer/input_layer_1/brand_embedding/brand_embedding_weights/embedding_lookup_sparseSparseSegmentMeansinput_layer/input_layer_1/brand_embedding/brand_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identitybinput_layer/input_layer_1/brand_embedding/brand_embedding_weights/embedding_lookup_sparse/Unique:1^input_layer/input_layer_1/brand_embedding/brand_embedding_weights/embedding_lookup_sparse/Cast*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
˘
Qinput_layer/input_layer_1/brand_embedding/brand_embedding_weights/Reshape_1/shapeConst*
dtype0*
valueB"˙˙˙˙   *
_output_shapes
:
Č
Kinput_layer/input_layer_1/brand_embedding/brand_embedding_weights/Reshape_1Reshapekinput_layer/input_layer_1/brand_embedding/brand_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:2Qinput_layer/input_layer_1/brand_embedding/brand_embedding_weights/Reshape_1/shape*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

Đ
Ginput_layer/input_layer_1/brand_embedding/brand_embedding_weights/ShapeShapeYinput_layer/input_layer_1/brand_embedding/brand_embedding_weights/embedding_lookup_sparse*
T0*
_output_shapes
:

Uinput_layer/input_layer_1/brand_embedding/brand_embedding_weights/strided_slice/stackConst*
valueB:*
dtype0*
_output_shapes
:
Ą
Winput_layer/input_layer_1/brand_embedding/brand_embedding_weights/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
Ą
Winput_layer/input_layer_1/brand_embedding/brand_embedding_weights/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
÷
Oinput_layer/input_layer_1/brand_embedding/brand_embedding_weights/strided_sliceStridedSliceGinput_layer/input_layer_1/brand_embedding/brand_embedding_weights/ShapeUinput_layer/input_layer_1/brand_embedding/brand_embedding_weights/strided_slice/stackWinput_layer/input_layer_1/brand_embedding/brand_embedding_weights/strided_slice/stack_1Winput_layer/input_layer_1/brand_embedding/brand_embedding_weights/strided_slice/stack_2*
T0*
Index0*
_output_shapes
: *
shrink_axis_mask

Iinput_layer/input_layer_1/brand_embedding/brand_embedding_weights/stack/0Const*
value	B :*
dtype0*
_output_shapes
: 

Ginput_layer/input_layer_1/brand_embedding/brand_embedding_weights/stackPackIinput_layer/input_layer_1/brand_embedding/brand_embedding_weights/stack/0Oinput_layer/input_layer_1/brand_embedding/brand_embedding_weights/strided_slice*
T0*
N*
_output_shapes
:

Finput_layer/input_layer_1/brand_embedding/brand_embedding_weights/TileTileKinput_layer/input_layer_1/brand_embedding/brand_embedding_weights/Reshape_1Ginput_layer/input_layer_1/brand_embedding/brand_embedding_weights/stack*
T0
*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
ć
Linput_layer/input_layer_1/brand_embedding/brand_embedding_weights/zeros_like	ZerosLikeYinput_layer/input_layer_1/brand_embedding/brand_embedding_weights/embedding_lookup_sparse*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
î
Ainput_layer/input_layer_1/brand_embedding/brand_embedding_weightsSelectFinput_layer/input_layer_1/brand_embedding/brand_embedding_weights/TileLinput_layer/input_layer_1/brand_embedding/brand_embedding_weights/zeros_likeYinput_layer/input_layer_1/brand_embedding/brand_embedding_weights/embedding_lookup_sparse*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ë
Hinput_layer/input_layer_1/brand_embedding/brand_embedding_weights/Cast_1CastEinput_layer/input_layer_1/brand_embedding/to_sparse_input/dense_shape*

SrcT0	*

DstT0*
_output_shapes
:

Oinput_layer/input_layer_1/brand_embedding/brand_embedding_weights/Slice_1/beginConst*
dtype0*
valueB: *
_output_shapes
:

Ninput_layer/input_layer_1/brand_embedding/brand_embedding_weights/Slice_1/sizeConst*
_output_shapes
:*
valueB:*
dtype0
ď
Iinput_layer/input_layer_1/brand_embedding/brand_embedding_weights/Slice_1SliceHinput_layer/input_layer_1/brand_embedding/brand_embedding_weights/Cast_1Oinput_layer/input_layer_1/brand_embedding/brand_embedding_weights/Slice_1/beginNinput_layer/input_layer_1/brand_embedding/brand_embedding_weights/Slice_1/size*
T0*
_output_shapes
:*
Index0
ş
Iinput_layer/input_layer_1/brand_embedding/brand_embedding_weights/Shape_1ShapeAinput_layer/input_layer_1/brand_embedding/brand_embedding_weights*
_output_shapes
:*
T0

Oinput_layer/input_layer_1/brand_embedding/brand_embedding_weights/Slice_2/beginConst*
valueB:*
_output_shapes
:*
dtype0
Ą
Ninput_layer/input_layer_1/brand_embedding/brand_embedding_weights/Slice_2/sizeConst*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0
đ
Iinput_layer/input_layer_1/brand_embedding/brand_embedding_weights/Slice_2SliceIinput_layer/input_layer_1/brand_embedding/brand_embedding_weights/Shape_1Oinput_layer/input_layer_1/brand_embedding/brand_embedding_weights/Slice_2/beginNinput_layer/input_layer_1/brand_embedding/brand_embedding_weights/Slice_2/size*
T0*
Index0*
_output_shapes
:

Minput_layer/input_layer_1/brand_embedding/brand_embedding_weights/concat/axisConst*
dtype0*
value	B : *
_output_shapes
: 
ç
Hinput_layer/input_layer_1/brand_embedding/brand_embedding_weights/concatConcatV2Iinput_layer/input_layer_1/brand_embedding/brand_embedding_weights/Slice_1Iinput_layer/input_layer_1/brand_embedding/brand_embedding_weights/Slice_2Minput_layer/input_layer_1/brand_embedding/brand_embedding_weights/concat/axis*
N*
_output_shapes
:*
T0

Kinput_layer/input_layer_1/brand_embedding/brand_embedding_weights/Reshape_2ReshapeAinput_layer/input_layer_1/brand_embedding/brand_embedding_weightsHinput_layer/input_layer_1/brand_embedding/brand_embedding_weights/concat*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ş
/input_layer/input_layer_1/brand_embedding/ShapeShapeKinput_layer/input_layer_1/brand_embedding/brand_embedding_weights/Reshape_2*
_output_shapes
:*
T0

=input_layer/input_layer_1/brand_embedding/strided_slice/stackConst*
dtype0*
valueB: *
_output_shapes
:

?input_layer/input_layer_1/brand_embedding/strided_slice/stack_1Const*
dtype0*
valueB:*
_output_shapes
:

?input_layer/input_layer_1/brand_embedding/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
˙
7input_layer/input_layer_1/brand_embedding/strided_sliceStridedSlice/input_layer/input_layer_1/brand_embedding/Shape=input_layer/input_layer_1/brand_embedding/strided_slice/stack?input_layer/input_layer_1/brand_embedding/strided_slice/stack_1?input_layer/input_layer_1/brand_embedding/strided_slice/stack_2*
T0*
shrink_axis_mask*
Index0*
_output_shapes
: 
{
9input_layer/input_layer_1/brand_embedding/Reshape/shape/1Const*
dtype0*
_output_shapes
: *
value	B :
á
7input_layer/input_layer_1/brand_embedding/Reshape/shapePack7input_layer/input_layer_1/brand_embedding/strided_slice9input_layer/input_layer_1/brand_embedding/Reshape/shape/1*
N*
_output_shapes
:*
T0
ô
1input_layer/input_layer_1/brand_embedding/ReshapeReshapeKinput_layer/input_layer_1/brand_embedding/brand_embedding_weights/Reshape_27input_layer/input_layer_1/brand_embedding/Reshape/shape*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

>input_layer/input_layer_1/campaign_id_embedding/ExpandDims/dimConst*
dtype0*
_output_shapes
: *
valueB :
˙˙˙˙˙˙˙˙˙
É
:input_layer/input_layer_1/campaign_id_embedding/ExpandDims
ExpandDimsPlaceholder_5>input_layer/input_layer_1/campaign_id_embedding/ExpandDims/dim*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

Ninput_layer/input_layer_1/campaign_id_embedding/to_sparse_input/ignore_value/xConst*
valueB B *
dtype0*
_output_shapes
: 

Hinput_layer/input_layer_1/campaign_id_embedding/to_sparse_input/NotEqualNotEqual:input_layer/input_layer_1/campaign_id_embedding/ExpandDimsNinput_layer/input_layer_1/campaign_id_embedding/to_sparse_input/ignore_value/x*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ă
Ginput_layer/input_layer_1/campaign_id_embedding/to_sparse_input/indicesWhereHinput_layer/input_layer_1/campaign_id_embedding/to_sparse_input/NotEqual*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

Finput_layer/input_layer_1/campaign_id_embedding/to_sparse_input/valuesGatherNd:input_layer/input_layer_1/campaign_id_embedding/ExpandDimsGinput_layer/input_layer_1/campaign_id_embedding/to_sparse_input/indices*
Tindices0	*
Tparams0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ĺ
Kinput_layer/input_layer_1/campaign_id_embedding/to_sparse_input/dense_shapeShape:input_layer/input_layer_1/campaign_id_embedding/ExpandDims*
out_type0	*
_output_shapes
:*
T0
Ň
6input_layer/input_layer_1/campaign_id_embedding/lookupStringToHashBucketFastFinput_layer/input_layer_1/campaign_id_embedding/to_sparse_input/values*
num_buckets *#
_output_shapes
:˙˙˙˙˙˙˙˙˙

kinput_layer/input_layer_1/campaign_id_embedding/embedding_weights/part_0/Initializer/truncated_normal/shapeConst*
dtype0*
_output_shapes
:*[
_classQ
OMloc:@input_layer/input_layer_1/campaign_id_embedding/embedding_weights/part_0*
valueB"     

jinput_layer/input_layer_1/campaign_id_embedding/embedding_weights/part_0/Initializer/truncated_normal/meanConst*
dtype0*
_output_shapes
: *
valueB
 *    *[
_classQ
OMloc:@input_layer/input_layer_1/campaign_id_embedding/embedding_weights/part_0

linput_layer/input_layer_1/campaign_id_embedding/embedding_weights/part_0/Initializer/truncated_normal/stddevConst*
dtype0*
_output_shapes
: *[
_classQ
OMloc:@input_layer/input_layer_1/campaign_id_embedding/embedding_weights/part_0*
valueB
 *  >

uinput_layer/input_layer_1/campaign_id_embedding/embedding_weights/part_0/Initializer/truncated_normal/TruncatedNormalTruncatedNormalkinput_layer/input_layer_1/campaign_id_embedding/embedding_weights/part_0/Initializer/truncated_normal/shape* 
_output_shapes
:
 *
T0*[
_classQ
OMloc:@input_layer/input_layer_1/campaign_id_embedding/embedding_weights/part_0*
dtype0
Ý
iinput_layer/input_layer_1/campaign_id_embedding/embedding_weights/part_0/Initializer/truncated_normal/mulMuluinput_layer/input_layer_1/campaign_id_embedding/embedding_weights/part_0/Initializer/truncated_normal/TruncatedNormallinput_layer/input_layer_1/campaign_id_embedding/embedding_weights/part_0/Initializer/truncated_normal/stddev*
T0* 
_output_shapes
:
 *[
_classQ
OMloc:@input_layer/input_layer_1/campaign_id_embedding/embedding_weights/part_0
Ë
einput_layer/input_layer_1/campaign_id_embedding/embedding_weights/part_0/Initializer/truncated_normalAddiinput_layer/input_layer_1/campaign_id_embedding/embedding_weights/part_0/Initializer/truncated_normal/muljinput_layer/input_layer_1/campaign_id_embedding/embedding_weights/part_0/Initializer/truncated_normal/mean* 
_output_shapes
:
 *[
_classQ
OMloc:@input_layer/input_layer_1/campaign_id_embedding/embedding_weights/part_0*
T0
ů
Hinput_layer/input_layer_1/campaign_id_embedding/embedding_weights/part_0
VariableV2*
shape:
 *
dtype0* 
_output_shapes
:
 *[
_classQ
OMloc:@input_layer/input_layer_1/campaign_id_embedding/embedding_weights/part_0

Oinput_layer/input_layer_1/campaign_id_embedding/embedding_weights/part_0/AssignAssignHinput_layer/input_layer_1/campaign_id_embedding/embedding_weights/part_0einput_layer/input_layer_1/campaign_id_embedding/embedding_weights/part_0/Initializer/truncated_normal*[
_classQ
OMloc:@input_layer/input_layer_1/campaign_id_embedding/embedding_weights/part_0* 
_output_shapes
:
 *
T0
Ť
Minput_layer/input_layer_1/campaign_id_embedding/embedding_weights/part_0/readIdentityHinput_layer/input_layer_1/campaign_id_embedding/embedding_weights/part_0* 
_output_shapes
:
 *[
_classQ
OMloc:@input_layer/input_layer_1/campaign_id_embedding/embedding_weights/part_0*
T0
Ł
Yinput_layer/input_layer_1/campaign_id_embedding/campaign_id_embedding_weights/Slice/beginConst*
dtype0*
valueB: *
_output_shapes
:
˘
Xinput_layer/input_layer_1/campaign_id_embedding/campaign_id_embedding_weights/Slice/sizeConst*
dtype0*
_output_shapes
:*
valueB:

Sinput_layer/input_layer_1/campaign_id_embedding/campaign_id_embedding_weights/SliceSliceKinput_layer/input_layer_1/campaign_id_embedding/to_sparse_input/dense_shapeYinput_layer/input_layer_1/campaign_id_embedding/campaign_id_embedding_weights/Slice/beginXinput_layer/input_layer_1/campaign_id_embedding/campaign_id_embedding_weights/Slice/size*
_output_shapes
:*
Index0*
T0	

Sinput_layer/input_layer_1/campaign_id_embedding/campaign_id_embedding_weights/ConstConst*
valueB: *
_output_shapes
:*
dtype0
Ľ
Rinput_layer/input_layer_1/campaign_id_embedding/campaign_id_embedding_weights/ProdProdSinput_layer/input_layer_1/campaign_id_embedding/campaign_id_embedding_weights/SliceSinput_layer/input_layer_1/campaign_id_embedding/campaign_id_embedding_weights/Const*
T0	*
_output_shapes
: 
 
^input_layer/input_layer_1/campaign_id_embedding/campaign_id_embedding_weights/GatherV2/indicesConst*
_output_shapes
: *
dtype0*
value	B :

[input_layer/input_layer_1/campaign_id_embedding/campaign_id_embedding_weights/GatherV2/axisConst*
dtype0*
value	B : *
_output_shapes
: 
°
Vinput_layer/input_layer_1/campaign_id_embedding/campaign_id_embedding_weights/GatherV2GatherV2Kinput_layer/input_layer_1/campaign_id_embedding/to_sparse_input/dense_shape^input_layer/input_layer_1/campaign_id_embedding/campaign_id_embedding_weights/GatherV2/indices[input_layer/input_layer_1/campaign_id_embedding/campaign_id_embedding_weights/GatherV2/axis*
Taxis0*
Tparams0	*
Tindices0*
_output_shapes
: 
ś
Tinput_layer/input_layer_1/campaign_id_embedding/campaign_id_embedding_weights/Cast/xPackRinput_layer/input_layer_1/campaign_id_embedding/campaign_id_embedding_weights/ProdVinput_layer/input_layer_1/campaign_id_embedding/campaign_id_embedding_weights/GatherV2*
_output_shapes
:*
T0	*
N

[input_layer/input_layer_1/campaign_id_embedding/campaign_id_embedding_weights/SparseReshapeSparseReshapeGinput_layer/input_layer_1/campaign_id_embedding/to_sparse_input/indicesKinput_layer/input_layer_1/campaign_id_embedding/to_sparse_input/dense_shapeTinput_layer/input_layer_1/campaign_id_embedding/campaign_id_embedding_weights/Cast/x*-
_output_shapes
:˙˙˙˙˙˙˙˙˙:
Ö
dinput_layer/input_layer_1/campaign_id_embedding/campaign_id_embedding_weights/SparseReshape/IdentityIdentity6input_layer/input_layer_1/campaign_id_embedding/lookup*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0	

\input_layer/input_layer_1/campaign_id_embedding/campaign_id_embedding_weights/GreaterEqual/yConst*
value	B	 R *
_output_shapes
: *
dtype0	
Ü
Zinput_layer/input_layer_1/campaign_id_embedding/campaign_id_embedding_weights/GreaterEqualGreaterEqualdinput_layer/input_layer_1/campaign_id_embedding/campaign_id_embedding_weights/SparseReshape/Identity\input_layer/input_layer_1/campaign_id_embedding/campaign_id_embedding_weights/GreaterEqual/y*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0	
á
Sinput_layer/input_layer_1/campaign_id_embedding/campaign_id_embedding_weights/WhereWhereZinput_layer/input_layer_1/campaign_id_embedding/campaign_id_embedding_weights/GreaterEqual*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ž
[input_layer/input_layer_1/campaign_id_embedding/campaign_id_embedding_weights/Reshape/shapeConst*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙*
_output_shapes
:
Ŕ
Uinput_layer/input_layer_1/campaign_id_embedding/campaign_id_embedding_weights/ReshapeReshapeSinput_layer/input_layer_1/campaign_id_embedding/campaign_id_embedding_weights/Where[input_layer/input_layer_1/campaign_id_embedding/campaign_id_embedding_weights/Reshape/shape*
T0	*#
_output_shapes
:˙˙˙˙˙˙˙˙˙

]input_layer/input_layer_1/campaign_id_embedding/campaign_id_embedding_weights/GatherV2_1/axisConst*
dtype0*
_output_shapes
: *
value	B : 
Ě
Xinput_layer/input_layer_1/campaign_id_embedding/campaign_id_embedding_weights/GatherV2_1GatherV2[input_layer/input_layer_1/campaign_id_embedding/campaign_id_embedding_weights/SparseReshapeUinput_layer/input_layer_1/campaign_id_embedding/campaign_id_embedding_weights/Reshape]input_layer/input_layer_1/campaign_id_embedding/campaign_id_embedding_weights/GatherV2_1/axis*
Taxis0*
Tindices0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
Tparams0	

]input_layer/input_layer_1/campaign_id_embedding/campaign_id_embedding_weights/GatherV2_2/axisConst*
value	B : *
_output_shapes
: *
dtype0
Ń
Xinput_layer/input_layer_1/campaign_id_embedding/campaign_id_embedding_weights/GatherV2_2GatherV2dinput_layer/input_layer_1/campaign_id_embedding/campaign_id_embedding_weights/SparseReshape/IdentityUinput_layer/input_layer_1/campaign_id_embedding/campaign_id_embedding_weights/Reshape]input_layer/input_layer_1/campaign_id_embedding/campaign_id_embedding_weights/GatherV2_2/axis*
Tparams0	*
Taxis0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
Tindices0	
ć
Vinput_layer/input_layer_1/campaign_id_embedding/campaign_id_embedding_weights/IdentityIdentity]input_layer/input_layer_1/campaign_id_embedding/campaign_id_embedding_weights/SparseReshape:1*
_output_shapes
:*
T0	
Š
ginput_layer/input_layer_1/campaign_id_embedding/campaign_id_embedding_weights/SparseFillEmptyRows/ConstConst*
dtype0	*
value	B	 R *
_output_shapes
: 
ŕ
uinput_layer/input_layer_1/campaign_id_embedding/campaign_id_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowsSparseFillEmptyRowsXinput_layer/input_layer_1/campaign_id_embedding/campaign_id_embedding_weights/GatherV2_1Xinput_layer/input_layer_1/campaign_id_embedding/campaign_id_embedding_weights/GatherV2_2Vinput_layer/input_layer_1/campaign_id_embedding/campaign_id_embedding_weights/Identityginput_layer/input_layer_1/campaign_id_embedding/campaign_id_embedding_weights/SparseFillEmptyRows/Const*T
_output_shapesB
@:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0	
Ę
yinput_layer/input_layer_1/campaign_id_embedding/campaign_id_embedding_weights/embedding_lookup_sparse/strided_slice/stackConst*
_output_shapes
:*
valueB"        *
dtype0
Ě
{input_layer/input_layer_1/campaign_id_embedding/campaign_id_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1Const*
valueB"       *
dtype0*
_output_shapes
:
Ě
{input_layer/input_layer_1/campaign_id_embedding/campaign_id_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB"      
ä
sinput_layer/input_layer_1/campaign_id_embedding/campaign_id_embedding_weights/embedding_lookup_sparse/strided_sliceStridedSliceuinput_layer/input_layer_1/campaign_id_embedding/campaign_id_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowsyinput_layer/input_layer_1/campaign_id_embedding/campaign_id_embedding_weights/embedding_lookup_sparse/strided_slice/stack{input_layer/input_layer_1/campaign_id_embedding/campaign_id_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1{input_layer/input_layer_1/campaign_id_embedding/campaign_id_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2*
Index0*
shrink_axis_mask*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0	*
end_mask*

begin_mask
¤
jinput_layer/input_layer_1/campaign_id_embedding/campaign_id_embedding_weights/embedding_lookup_sparse/CastCastsinput_layer/input_layer_1/campaign_id_embedding/campaign_id_embedding_weights/embedding_lookup_sparse/strided_slice*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*

SrcT0	*

DstT0
Ź
linput_layer/input_layer_1/campaign_id_embedding/campaign_id_embedding_weights/embedding_lookup_sparse/UniqueUniquewinput_layer/input_layer_1/campaign_id_embedding/campaign_id_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0	

{input_layer/input_layer_1/campaign_id_embedding/campaign_id_embedding_weights/embedding_lookup_sparse/embedding_lookup/axisConst*
dtype0*[
_classQ
OMloc:@input_layer/input_layer_1/campaign_id_embedding/embedding_weights/part_0*
_output_shapes
: *
value	B : 
î
vinput_layer/input_layer_1/campaign_id_embedding/campaign_id_embedding_weights/embedding_lookup_sparse/embedding_lookupGatherV2Minput_layer/input_layer_1/campaign_id_embedding/embedding_weights/part_0/readlinput_layer/input_layer_1/campaign_id_embedding/campaign_id_embedding_weights/embedding_lookup_sparse/Unique{input_layer/input_layer_1/campaign_id_embedding/campaign_id_embedding_weights/embedding_lookup_sparse/embedding_lookup/axis*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
Taxis0*
Tindices0	*
Tparams0*[
_classQ
OMloc:@input_layer/input_layer_1/campaign_id_embedding/embedding_weights/part_0
ľ
input_layer/input_layer_1/campaign_id_embedding/campaign_id_embedding_weights/embedding_lookup_sparse/embedding_lookup/IdentityIdentityvinput_layer/input_layer_1/campaign_id_embedding/campaign_id_embedding_weights/embedding_lookup_sparse/embedding_lookup*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

einput_layer/input_layer_1/campaign_id_embedding/campaign_id_embedding_weights/embedding_lookup_sparseSparseSegmentMeaninput_layer/input_layer_1/campaign_id_embedding/campaign_id_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identityninput_layer/input_layer_1/campaign_id_embedding/campaign_id_embedding_weights/embedding_lookup_sparse/Unique:1jinput_layer/input_layer_1/campaign_id_embedding/campaign_id_embedding_weights/embedding_lookup_sparse/Cast*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ž
]input_layer/input_layer_1/campaign_id_embedding/campaign_id_embedding_weights/Reshape_1/shapeConst*
valueB"˙˙˙˙   *
dtype0*
_output_shapes
:
ě
Winput_layer/input_layer_1/campaign_id_embedding/campaign_id_embedding_weights/Reshape_1Reshapewinput_layer/input_layer_1/campaign_id_embedding/campaign_id_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:2]input_layer/input_layer_1/campaign_id_embedding/campaign_id_embedding_weights/Reshape_1/shape*
T0
*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
č
Sinput_layer/input_layer_1/campaign_id_embedding/campaign_id_embedding_weights/ShapeShapeeinput_layer/input_layer_1/campaign_id_embedding/campaign_id_embedding_weights/embedding_lookup_sparse*
_output_shapes
:*
T0
Ť
ainput_layer/input_layer_1/campaign_id_embedding/campaign_id_embedding_weights/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB:
­
cinput_layer/input_layer_1/campaign_id_embedding/campaign_id_embedding_weights/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
­
cinput_layer/input_layer_1/campaign_id_embedding/campaign_id_embedding_weights/strided_slice/stack_2Const*
valueB:*
_output_shapes
:*
dtype0
ł
[input_layer/input_layer_1/campaign_id_embedding/campaign_id_embedding_weights/strided_sliceStridedSliceSinput_layer/input_layer_1/campaign_id_embedding/campaign_id_embedding_weights/Shapeainput_layer/input_layer_1/campaign_id_embedding/campaign_id_embedding_weights/strided_slice/stackcinput_layer/input_layer_1/campaign_id_embedding/campaign_id_embedding_weights/strided_slice/stack_1cinput_layer/input_layer_1/campaign_id_embedding/campaign_id_embedding_weights/strided_slice/stack_2*
T0*
Index0*
_output_shapes
: *
shrink_axis_mask

Uinput_layer/input_layer_1/campaign_id_embedding/campaign_id_embedding_weights/stack/0Const*
value	B :*
dtype0*
_output_shapes
: 
˝
Sinput_layer/input_layer_1/campaign_id_embedding/campaign_id_embedding_weights/stackPackUinput_layer/input_layer_1/campaign_id_embedding/campaign_id_embedding_weights/stack/0[input_layer/input_layer_1/campaign_id_embedding/campaign_id_embedding_weights/strided_slice*
N*
T0*
_output_shapes
:
Ă
Rinput_layer/input_layer_1/campaign_id_embedding/campaign_id_embedding_weights/TileTileWinput_layer/input_layer_1/campaign_id_embedding/campaign_id_embedding_weights/Reshape_1Sinput_layer/input_layer_1/campaign_id_embedding/campaign_id_embedding_weights/stack*
T0
*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
ţ
Xinput_layer/input_layer_1/campaign_id_embedding/campaign_id_embedding_weights/zeros_like	ZerosLikeeinput_layer/input_layer_1/campaign_id_embedding/campaign_id_embedding_weights/embedding_lookup_sparse*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

Minput_layer/input_layer_1/campaign_id_embedding/campaign_id_embedding_weightsSelectRinput_layer/input_layer_1/campaign_id_embedding/campaign_id_embedding_weights/TileXinput_layer/input_layer_1/campaign_id_embedding/campaign_id_embedding_weights/zeros_likeeinput_layer/input_layer_1/campaign_id_embedding/campaign_id_embedding_weights/embedding_lookup_sparse*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ý
Tinput_layer/input_layer_1/campaign_id_embedding/campaign_id_embedding_weights/Cast_1CastKinput_layer/input_layer_1/campaign_id_embedding/to_sparse_input/dense_shape*

SrcT0	*
_output_shapes
:*

DstT0
Ľ
[input_layer/input_layer_1/campaign_id_embedding/campaign_id_embedding_weights/Slice_1/beginConst*
_output_shapes
:*
dtype0*
valueB: 
¤
Zinput_layer/input_layer_1/campaign_id_embedding/campaign_id_embedding_weights/Slice_1/sizeConst*
valueB:*
_output_shapes
:*
dtype0

Uinput_layer/input_layer_1/campaign_id_embedding/campaign_id_embedding_weights/Slice_1SliceTinput_layer/input_layer_1/campaign_id_embedding/campaign_id_embedding_weights/Cast_1[input_layer/input_layer_1/campaign_id_embedding/campaign_id_embedding_weights/Slice_1/beginZinput_layer/input_layer_1/campaign_id_embedding/campaign_id_embedding_weights/Slice_1/size*
T0*
_output_shapes
:*
Index0
Ň
Uinput_layer/input_layer_1/campaign_id_embedding/campaign_id_embedding_weights/Shape_1ShapeMinput_layer/input_layer_1/campaign_id_embedding/campaign_id_embedding_weights*
_output_shapes
:*
T0
Ľ
[input_layer/input_layer_1/campaign_id_embedding/campaign_id_embedding_weights/Slice_2/beginConst*
valueB:*
_output_shapes
:*
dtype0
­
Zinput_layer/input_layer_1/campaign_id_embedding/campaign_id_embedding_weights/Slice_2/sizeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
 
Uinput_layer/input_layer_1/campaign_id_embedding/campaign_id_embedding_weights/Slice_2SliceUinput_layer/input_layer_1/campaign_id_embedding/campaign_id_embedding_weights/Shape_1[input_layer/input_layer_1/campaign_id_embedding/campaign_id_embedding_weights/Slice_2/beginZinput_layer/input_layer_1/campaign_id_embedding/campaign_id_embedding_weights/Slice_2/size*
T0*
Index0*
_output_shapes
:

Yinput_layer/input_layer_1/campaign_id_embedding/campaign_id_embedding_weights/concat/axisConst*
_output_shapes
: *
value	B : *
dtype0

Tinput_layer/input_layer_1/campaign_id_embedding/campaign_id_embedding_weights/concatConcatV2Uinput_layer/input_layer_1/campaign_id_embedding/campaign_id_embedding_weights/Slice_1Uinput_layer/input_layer_1/campaign_id_embedding/campaign_id_embedding_weights/Slice_2Yinput_layer/input_layer_1/campaign_id_embedding/campaign_id_embedding_weights/concat/axis*
N*
T0*
_output_shapes
:
š
Winput_layer/input_layer_1/campaign_id_embedding/campaign_id_embedding_weights/Reshape_2ReshapeMinput_layer/input_layer_1/campaign_id_embedding/campaign_id_embedding_weightsTinput_layer/input_layer_1/campaign_id_embedding/campaign_id_embedding_weights/concat*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
ź
5input_layer/input_layer_1/campaign_id_embedding/ShapeShapeWinput_layer/input_layer_1/campaign_id_embedding/campaign_id_embedding_weights/Reshape_2*
_output_shapes
:*
T0

Cinput_layer/input_layer_1/campaign_id_embedding/strided_slice/stackConst*
valueB: *
_output_shapes
:*
dtype0

Einput_layer/input_layer_1/campaign_id_embedding/strided_slice/stack_1Const*
valueB:*
_output_shapes
:*
dtype0

Einput_layer/input_layer_1/campaign_id_embedding/strided_slice/stack_2Const*
dtype0*
valueB:*
_output_shapes
:

=input_layer/input_layer_1/campaign_id_embedding/strided_sliceStridedSlice5input_layer/input_layer_1/campaign_id_embedding/ShapeCinput_layer/input_layer_1/campaign_id_embedding/strided_slice/stackEinput_layer/input_layer_1/campaign_id_embedding/strided_slice/stack_1Einput_layer/input_layer_1/campaign_id_embedding/strided_slice/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask

?input_layer/input_layer_1/campaign_id_embedding/Reshape/shape/1Const*
dtype0*
value	B :*
_output_shapes
: 
ó
=input_layer/input_layer_1/campaign_id_embedding/Reshape/shapePack=input_layer/input_layer_1/campaign_id_embedding/strided_slice?input_layer/input_layer_1/campaign_id_embedding/Reshape/shape/1*
N*
T0*
_output_shapes
:

7input_layer/input_layer_1/campaign_id_embedding/ReshapeReshapeWinput_layer/input_layer_1/campaign_id_embedding/campaign_id_embedding_weights/Reshape_2=input_layer/input_layer_1/campaign_id_embedding/Reshape/shape*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

:input_layer/input_layer_1/cate_id_embedding/ExpandDims/dimConst*
dtype0*
_output_shapes
: *
valueB :
˙˙˙˙˙˙˙˙˙
Á
6input_layer/input_layer_1/cate_id_embedding/ExpandDims
ExpandDimsPlaceholder_4:input_layer/input_layer_1/cate_id_embedding/ExpandDims/dim*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

Jinput_layer/input_layer_1/cate_id_embedding/to_sparse_input/ignore_value/xConst*
valueB B *
_output_shapes
: *
dtype0

Dinput_layer/input_layer_1/cate_id_embedding/to_sparse_input/NotEqualNotEqual6input_layer/input_layer_1/cate_id_embedding/ExpandDimsJinput_layer/input_layer_1/cate_id_embedding/to_sparse_input/ignore_value/x*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
ť
Cinput_layer/input_layer_1/cate_id_embedding/to_sparse_input/indicesWhereDinput_layer/input_layer_1/cate_id_embedding/to_sparse_input/NotEqual*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

Binput_layer/input_layer_1/cate_id_embedding/to_sparse_input/valuesGatherNd6input_layer/input_layer_1/cate_id_embedding/ExpandDimsCinput_layer/input_layer_1/cate_id_embedding/to_sparse_input/indices*
Tindices0	*
Tparams0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
˝
Ginput_layer/input_layer_1/cate_id_embedding/to_sparse_input/dense_shapeShape6input_layer/input_layer_1/cate_id_embedding/ExpandDims*
out_type0	*
T0*
_output_shapes
:
É
2input_layer/input_layer_1/cate_id_embedding/lookupStringToHashBucketFastBinput_layer/input_layer_1/cate_id_embedding/to_sparse_input/values*
num_bucketsN*#
_output_shapes
:˙˙˙˙˙˙˙˙˙

ginput_layer/input_layer_1/cate_id_embedding/embedding_weights/part_0/Initializer/truncated_normal/shapeConst*
valueB"'     *
dtype0*
_output_shapes
:*W
_classM
KIloc:@input_layer/input_layer_1/cate_id_embedding/embedding_weights/part_0

finput_layer/input_layer_1/cate_id_embedding/embedding_weights/part_0/Initializer/truncated_normal/meanConst*
valueB
 *    *
_output_shapes
: *
dtype0*W
_classM
KIloc:@input_layer/input_layer_1/cate_id_embedding/embedding_weights/part_0

hinput_layer/input_layer_1/cate_id_embedding/embedding_weights/part_0/Initializer/truncated_normal/stddevConst*
valueB
 *  >*
dtype0*W
_classM
KIloc:@input_layer/input_layer_1/cate_id_embedding/embedding_weights/part_0*
_output_shapes
: 
ý
qinput_layer/input_layer_1/cate_id_embedding/embedding_weights/part_0/Initializer/truncated_normal/TruncatedNormalTruncatedNormalginput_layer/input_layer_1/cate_id_embedding/embedding_weights/part_0/Initializer/truncated_normal/shape*
T0*
_output_shapes
:	N*
dtype0*W
_classM
KIloc:@input_layer/input_layer_1/cate_id_embedding/embedding_weights/part_0
Ě
einput_layer/input_layer_1/cate_id_embedding/embedding_weights/part_0/Initializer/truncated_normal/mulMulqinput_layer/input_layer_1/cate_id_embedding/embedding_weights/part_0/Initializer/truncated_normal/TruncatedNormalhinput_layer/input_layer_1/cate_id_embedding/embedding_weights/part_0/Initializer/truncated_normal/stddev*
_output_shapes
:	N*W
_classM
KIloc:@input_layer/input_layer_1/cate_id_embedding/embedding_weights/part_0*
T0
ş
ainput_layer/input_layer_1/cate_id_embedding/embedding_weights/part_0/Initializer/truncated_normalAddeinput_layer/input_layer_1/cate_id_embedding/embedding_weights/part_0/Initializer/truncated_normal/mulfinput_layer/input_layer_1/cate_id_embedding/embedding_weights/part_0/Initializer/truncated_normal/mean*
T0*
_output_shapes
:	N*W
_classM
KIloc:@input_layer/input_layer_1/cate_id_embedding/embedding_weights/part_0
ď
Dinput_layer/input_layer_1/cate_id_embedding/embedding_weights/part_0
VariableV2*
_output_shapes
:	N*
shape:	N*W
_classM
KIloc:@input_layer/input_layer_1/cate_id_embedding/embedding_weights/part_0*
dtype0

Kinput_layer/input_layer_1/cate_id_embedding/embedding_weights/part_0/AssignAssignDinput_layer/input_layer_1/cate_id_embedding/embedding_weights/part_0ainput_layer/input_layer_1/cate_id_embedding/embedding_weights/part_0/Initializer/truncated_normal*W
_classM
KIloc:@input_layer/input_layer_1/cate_id_embedding/embedding_weights/part_0*
T0*
_output_shapes
:	N

Iinput_layer/input_layer_1/cate_id_embedding/embedding_weights/part_0/readIdentityDinput_layer/input_layer_1/cate_id_embedding/embedding_weights/part_0*W
_classM
KIloc:@input_layer/input_layer_1/cate_id_embedding/embedding_weights/part_0*
_output_shapes
:	N*
T0

Qinput_layer/input_layer_1/cate_id_embedding/cate_id_embedding_weights/Slice/beginConst*
valueB: *
_output_shapes
:*
dtype0

Pinput_layer/input_layer_1/cate_id_embedding/cate_id_embedding_weights/Slice/sizeConst*
_output_shapes
:*
valueB:*
dtype0
ô
Kinput_layer/input_layer_1/cate_id_embedding/cate_id_embedding_weights/SliceSliceGinput_layer/input_layer_1/cate_id_embedding/to_sparse_input/dense_shapeQinput_layer/input_layer_1/cate_id_embedding/cate_id_embedding_weights/Slice/beginPinput_layer/input_layer_1/cate_id_embedding/cate_id_embedding_weights/Slice/size*
_output_shapes
:*
Index0*
T0	

Kinput_layer/input_layer_1/cate_id_embedding/cate_id_embedding_weights/ConstConst*
dtype0*
_output_shapes
:*
valueB: 

Jinput_layer/input_layer_1/cate_id_embedding/cate_id_embedding_weights/ProdProdKinput_layer/input_layer_1/cate_id_embedding/cate_id_embedding_weights/SliceKinput_layer/input_layer_1/cate_id_embedding/cate_id_embedding_weights/Const*
_output_shapes
: *
T0	

Vinput_layer/input_layer_1/cate_id_embedding/cate_id_embedding_weights/GatherV2/indicesConst*
value	B :*
dtype0*
_output_shapes
: 

Sinput_layer/input_layer_1/cate_id_embedding/cate_id_embedding_weights/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 

Ninput_layer/input_layer_1/cate_id_embedding/cate_id_embedding_weights/GatherV2GatherV2Ginput_layer/input_layer_1/cate_id_embedding/to_sparse_input/dense_shapeVinput_layer/input_layer_1/cate_id_embedding/cate_id_embedding_weights/GatherV2/indicesSinput_layer/input_layer_1/cate_id_embedding/cate_id_embedding_weights/GatherV2/axis*
Tindices0*
Tparams0	*
Taxis0*
_output_shapes
: 

Linput_layer/input_layer_1/cate_id_embedding/cate_id_embedding_weights/Cast/xPackJinput_layer/input_layer_1/cate_id_embedding/cate_id_embedding_weights/ProdNinput_layer/input_layer_1/cate_id_embedding/cate_id_embedding_weights/GatherV2*
_output_shapes
:*
T0	*
N
ď
Sinput_layer/input_layer_1/cate_id_embedding/cate_id_embedding_weights/SparseReshapeSparseReshapeCinput_layer/input_layer_1/cate_id_embedding/to_sparse_input/indicesGinput_layer/input_layer_1/cate_id_embedding/to_sparse_input/dense_shapeLinput_layer/input_layer_1/cate_id_embedding/cate_id_embedding_weights/Cast/x*-
_output_shapes
:˙˙˙˙˙˙˙˙˙:
Ę
\input_layer/input_layer_1/cate_id_embedding/cate_id_embedding_weights/SparseReshape/IdentityIdentity2input_layer/input_layer_1/cate_id_embedding/lookup*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0	

Tinput_layer/input_layer_1/cate_id_embedding/cate_id_embedding_weights/GreaterEqual/yConst*
_output_shapes
: *
value	B	 R *
dtype0	
Ä
Rinput_layer/input_layer_1/cate_id_embedding/cate_id_embedding_weights/GreaterEqualGreaterEqual\input_layer/input_layer_1/cate_id_embedding/cate_id_embedding_weights/SparseReshape/IdentityTinput_layer/input_layer_1/cate_id_embedding/cate_id_embedding_weights/GreaterEqual/y*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0	
Ń
Kinput_layer/input_layer_1/cate_id_embedding/cate_id_embedding_weights/WhereWhereRinput_layer/input_layer_1/cate_id_embedding/cate_id_embedding_weights/GreaterEqual*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ś
Sinput_layer/input_layer_1/cate_id_embedding/cate_id_embedding_weights/Reshape/shapeConst*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0
¨
Minput_layer/input_layer_1/cate_id_embedding/cate_id_embedding_weights/ReshapeReshapeKinput_layer/input_layer_1/cate_id_embedding/cate_id_embedding_weights/WhereSinput_layer/input_layer_1/cate_id_embedding/cate_id_embedding_weights/Reshape/shape*
T0	*#
_output_shapes
:˙˙˙˙˙˙˙˙˙

Uinput_layer/input_layer_1/cate_id_embedding/cate_id_embedding_weights/GatherV2_1/axisConst*
dtype0*
_output_shapes
: *
value	B : 
Ź
Pinput_layer/input_layer_1/cate_id_embedding/cate_id_embedding_weights/GatherV2_1GatherV2Sinput_layer/input_layer_1/cate_id_embedding/cate_id_embedding_weights/SparseReshapeMinput_layer/input_layer_1/cate_id_embedding/cate_id_embedding_weights/ReshapeUinput_layer/input_layer_1/cate_id_embedding/cate_id_embedding_weights/GatherV2_1/axis*
Taxis0*
Tparams0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
Tindices0	

Uinput_layer/input_layer_1/cate_id_embedding/cate_id_embedding_weights/GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
value	B : 
ą
Pinput_layer/input_layer_1/cate_id_embedding/cate_id_embedding_weights/GatherV2_2GatherV2\input_layer/input_layer_1/cate_id_embedding/cate_id_embedding_weights/SparseReshape/IdentityMinput_layer/input_layer_1/cate_id_embedding/cate_id_embedding_weights/ReshapeUinput_layer/input_layer_1/cate_id_embedding/cate_id_embedding_weights/GatherV2_2/axis*
Tindices0	*
Tparams0	*
Taxis0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ö
Ninput_layer/input_layer_1/cate_id_embedding/cate_id_embedding_weights/IdentityIdentityUinput_layer/input_layer_1/cate_id_embedding/cate_id_embedding_weights/SparseReshape:1*
T0	*
_output_shapes
:
Ą
_input_layer/input_layer_1/cate_id_embedding/cate_id_embedding_weights/SparseFillEmptyRows/ConstConst*
value	B	 R *
_output_shapes
: *
dtype0	
¸
minput_layer/input_layer_1/cate_id_embedding/cate_id_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowsSparseFillEmptyRowsPinput_layer/input_layer_1/cate_id_embedding/cate_id_embedding_weights/GatherV2_1Pinput_layer/input_layer_1/cate_id_embedding/cate_id_embedding_weights/GatherV2_2Ninput_layer/input_layer_1/cate_id_embedding/cate_id_embedding_weights/Identity_input_layer/input_layer_1/cate_id_embedding/cate_id_embedding_weights/SparseFillEmptyRows/Const*
T0	*T
_output_shapesB
@:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
Â
qinput_layer/input_layer_1/cate_id_embedding/cate_id_embedding_weights/embedding_lookup_sparse/strided_slice/stackConst*
dtype0*
valueB"        *
_output_shapes
:
Ä
sinput_layer/input_layer_1/cate_id_embedding/cate_id_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1Const*
valueB"       *
_output_shapes
:*
dtype0
Ä
sinput_layer/input_layer_1/cate_id_embedding/cate_id_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
ź
kinput_layer/input_layer_1/cate_id_embedding/cate_id_embedding_weights/embedding_lookup_sparse/strided_sliceStridedSliceminput_layer/input_layer_1/cate_id_embedding/cate_id_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowsqinput_layer/input_layer_1/cate_id_embedding/cate_id_embedding_weights/embedding_lookup_sparse/strided_slice/stacksinput_layer/input_layer_1/cate_id_embedding/cate_id_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1sinput_layer/input_layer_1/cate_id_embedding/cate_id_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2*
T0	*

begin_mask*
Index0*
end_mask*
shrink_axis_mask*#
_output_shapes
:˙˙˙˙˙˙˙˙˙

binput_layer/input_layer_1/cate_id_embedding/cate_id_embedding_weights/embedding_lookup_sparse/CastCastkinput_layer/input_layer_1/cate_id_embedding/cate_id_embedding_weights/embedding_lookup_sparse/strided_slice*

SrcT0	*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*

DstT0

dinput_layer/input_layer_1/cate_id_embedding/cate_id_embedding_weights/embedding_lookup_sparse/UniqueUniqueoinput_layer/input_layer_1/cate_id_embedding/cate_id_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:1*
T0	*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙

sinput_layer/input_layer_1/cate_id_embedding/cate_id_embedding_weights/embedding_lookup_sparse/embedding_lookup/axisConst*
dtype0*W
_classM
KIloc:@input_layer/input_layer_1/cate_id_embedding/embedding_weights/part_0*
_output_shapes
: *
value	B : 
Î
ninput_layer/input_layer_1/cate_id_embedding/cate_id_embedding_weights/embedding_lookup_sparse/embedding_lookupGatherV2Iinput_layer/input_layer_1/cate_id_embedding/embedding_weights/part_0/readdinput_layer/input_layer_1/cate_id_embedding/cate_id_embedding_weights/embedding_lookup_sparse/Uniquesinput_layer/input_layer_1/cate_id_embedding/cate_id_embedding_weights/embedding_lookup_sparse/embedding_lookup/axis*
Taxis0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*W
_classM
KIloc:@input_layer/input_layer_1/cate_id_embedding/embedding_weights/part_0*
Tparams0*
Tindices0	
Ľ
winput_layer/input_layer_1/cate_id_embedding/cate_id_embedding_weights/embedding_lookup_sparse/embedding_lookup/IdentityIdentityninput_layer/input_layer_1/cate_id_embedding/cate_id_embedding_weights/embedding_lookup_sparse/embedding_lookup*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
é
]input_layer/input_layer_1/cate_id_embedding/cate_id_embedding_weights/embedding_lookup_sparseSparseSegmentMeanwinput_layer/input_layer_1/cate_id_embedding/cate_id_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identityfinput_layer/input_layer_1/cate_id_embedding/cate_id_embedding_weights/embedding_lookup_sparse/Unique:1binput_layer/input_layer_1/cate_id_embedding/cate_id_embedding_weights/embedding_lookup_sparse/Cast*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ś
Uinput_layer/input_layer_1/cate_id_embedding/cate_id_embedding_weights/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   
Ô
Oinput_layer/input_layer_1/cate_id_embedding/cate_id_embedding_weights/Reshape_1Reshapeoinput_layer/input_layer_1/cate_id_embedding/cate_id_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:2Uinput_layer/input_layer_1/cate_id_embedding/cate_id_embedding_weights/Reshape_1/shape*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

Ř
Kinput_layer/input_layer_1/cate_id_embedding/cate_id_embedding_weights/ShapeShape]input_layer/input_layer_1/cate_id_embedding/cate_id_embedding_weights/embedding_lookup_sparse*
_output_shapes
:*
T0
Ł
Yinput_layer/input_layer_1/cate_id_embedding/cate_id_embedding_weights/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB:
Ľ
[input_layer/input_layer_1/cate_id_embedding/cate_id_embedding_weights/strided_slice/stack_1Const*
_output_shapes
:*
valueB:*
dtype0
Ľ
[input_layer/input_layer_1/cate_id_embedding/cate_id_embedding_weights/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:

Sinput_layer/input_layer_1/cate_id_embedding/cate_id_embedding_weights/strided_sliceStridedSliceKinput_layer/input_layer_1/cate_id_embedding/cate_id_embedding_weights/ShapeYinput_layer/input_layer_1/cate_id_embedding/cate_id_embedding_weights/strided_slice/stack[input_layer/input_layer_1/cate_id_embedding/cate_id_embedding_weights/strided_slice/stack_1[input_layer/input_layer_1/cate_id_embedding/cate_id_embedding_weights/strided_slice/stack_2*
T0*
_output_shapes
: *
shrink_axis_mask*
Index0

Minput_layer/input_layer_1/cate_id_embedding/cate_id_embedding_weights/stack/0Const*
dtype0*
_output_shapes
: *
value	B :
Ľ
Kinput_layer/input_layer_1/cate_id_embedding/cate_id_embedding_weights/stackPackMinput_layer/input_layer_1/cate_id_embedding/cate_id_embedding_weights/stack/0Sinput_layer/input_layer_1/cate_id_embedding/cate_id_embedding_weights/strided_slice*
N*
_output_shapes
:*
T0
Ť
Jinput_layer/input_layer_1/cate_id_embedding/cate_id_embedding_weights/TileTileOinput_layer/input_layer_1/cate_id_embedding/cate_id_embedding_weights/Reshape_1Kinput_layer/input_layer_1/cate_id_embedding/cate_id_embedding_weights/stack*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
T0

î
Pinput_layer/input_layer_1/cate_id_embedding/cate_id_embedding_weights/zeros_like	ZerosLike]input_layer/input_layer_1/cate_id_embedding/cate_id_embedding_weights/embedding_lookup_sparse*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
ţ
Einput_layer/input_layer_1/cate_id_embedding/cate_id_embedding_weightsSelectJinput_layer/input_layer_1/cate_id_embedding/cate_id_embedding_weights/TilePinput_layer/input_layer_1/cate_id_embedding/cate_id_embedding_weights/zeros_like]input_layer/input_layer_1/cate_id_embedding/cate_id_embedding_weights/embedding_lookup_sparse*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ń
Linput_layer/input_layer_1/cate_id_embedding/cate_id_embedding_weights/Cast_1CastGinput_layer/input_layer_1/cate_id_embedding/to_sparse_input/dense_shape*

DstT0*
_output_shapes
:*

SrcT0	

Sinput_layer/input_layer_1/cate_id_embedding/cate_id_embedding_weights/Slice_1/beginConst*
_output_shapes
:*
valueB: *
dtype0

Rinput_layer/input_layer_1/cate_id_embedding/cate_id_embedding_weights/Slice_1/sizeConst*
_output_shapes
:*
dtype0*
valueB:
˙
Minput_layer/input_layer_1/cate_id_embedding/cate_id_embedding_weights/Slice_1SliceLinput_layer/input_layer_1/cate_id_embedding/cate_id_embedding_weights/Cast_1Sinput_layer/input_layer_1/cate_id_embedding/cate_id_embedding_weights/Slice_1/beginRinput_layer/input_layer_1/cate_id_embedding/cate_id_embedding_weights/Slice_1/size*
Index0*
_output_shapes
:*
T0
Â
Minput_layer/input_layer_1/cate_id_embedding/cate_id_embedding_weights/Shape_1ShapeEinput_layer/input_layer_1/cate_id_embedding/cate_id_embedding_weights*
T0*
_output_shapes
:

Sinput_layer/input_layer_1/cate_id_embedding/cate_id_embedding_weights/Slice_2/beginConst*
_output_shapes
:*
dtype0*
valueB:
Ľ
Rinput_layer/input_layer_1/cate_id_embedding/cate_id_embedding_weights/Slice_2/sizeConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙

Minput_layer/input_layer_1/cate_id_embedding/cate_id_embedding_weights/Slice_2SliceMinput_layer/input_layer_1/cate_id_embedding/cate_id_embedding_weights/Shape_1Sinput_layer/input_layer_1/cate_id_embedding/cate_id_embedding_weights/Slice_2/beginRinput_layer/input_layer_1/cate_id_embedding/cate_id_embedding_weights/Slice_2/size*
Index0*
T0*
_output_shapes
:

Qinput_layer/input_layer_1/cate_id_embedding/cate_id_embedding_weights/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
÷
Linput_layer/input_layer_1/cate_id_embedding/cate_id_embedding_weights/concatConcatV2Minput_layer/input_layer_1/cate_id_embedding/cate_id_embedding_weights/Slice_1Minput_layer/input_layer_1/cate_id_embedding/cate_id_embedding_weights/Slice_2Qinput_layer/input_layer_1/cate_id_embedding/cate_id_embedding_weights/concat/axis*
_output_shapes
:*
T0*
N
Ą
Oinput_layer/input_layer_1/cate_id_embedding/cate_id_embedding_weights/Reshape_2ReshapeEinput_layer/input_layer_1/cate_id_embedding/cate_id_embedding_weightsLinput_layer/input_layer_1/cate_id_embedding/cate_id_embedding_weights/concat*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
°
1input_layer/input_layer_1/cate_id_embedding/ShapeShapeOinput_layer/input_layer_1/cate_id_embedding/cate_id_embedding_weights/Reshape_2*
T0*
_output_shapes
:

?input_layer/input_layer_1/cate_id_embedding/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:

Ainput_layer/input_layer_1/cate_id_embedding/strided_slice/stack_1Const*
dtype0*
valueB:*
_output_shapes
:

Ainput_layer/input_layer_1/cate_id_embedding/strided_slice/stack_2Const*
_output_shapes
:*
valueB:*
dtype0

9input_layer/input_layer_1/cate_id_embedding/strided_sliceStridedSlice1input_layer/input_layer_1/cate_id_embedding/Shape?input_layer/input_layer_1/cate_id_embedding/strided_slice/stackAinput_layer/input_layer_1/cate_id_embedding/strided_slice/stack_1Ainput_layer/input_layer_1/cate_id_embedding/strided_slice/stack_2*
shrink_axis_mask*
Index0*
T0*
_output_shapes
: 
}
;input_layer/input_layer_1/cate_id_embedding/Reshape/shape/1Const*
dtype0*
value	B :*
_output_shapes
: 
ç
9input_layer/input_layer_1/cate_id_embedding/Reshape/shapePack9input_layer/input_layer_1/cate_id_embedding/strided_slice;input_layer/input_layer_1/cate_id_embedding/Reshape/shape/1*
T0*
_output_shapes
:*
N
ü
3input_layer/input_layer_1/cate_id_embedding/ReshapeReshapeOinput_layer/input_layer_1/cate_id_embedding/cate_id_embedding_weights/Reshape_29input_layer/input_layer_1/cate_id_embedding/Reshape/shape*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

;input_layer/input_layer_1/customer_embedding/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙
Ă
7input_layer/input_layer_1/customer_embedding/ExpandDims
ExpandDimsPlaceholder_6;input_layer/input_layer_1/customer_embedding/ExpandDims/dim*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

Kinput_layer/input_layer_1/customer_embedding/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB B 

Einput_layer/input_layer_1/customer_embedding/to_sparse_input/NotEqualNotEqual7input_layer/input_layer_1/customer_embedding/ExpandDimsKinput_layer/input_layer_1/customer_embedding/to_sparse_input/ignore_value/x*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
˝
Dinput_layer/input_layer_1/customer_embedding/to_sparse_input/indicesWhereEinput_layer/input_layer_1/customer_embedding/to_sparse_input/NotEqual*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

Cinput_layer/input_layer_1/customer_embedding/to_sparse_input/valuesGatherNd7input_layer/input_layer_1/customer_embedding/ExpandDimsDinput_layer/input_layer_1/customer_embedding/to_sparse_input/indices*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
Tparams0*
Tindices0	
ż
Hinput_layer/input_layer_1/customer_embedding/to_sparse_input/dense_shapeShape7input_layer/input_layer_1/customer_embedding/ExpandDims*
_output_shapes
:*
T0*
out_type0	
Ě
3input_layer/input_layer_1/customer_embedding/lookupStringToHashBucketFastCinput_layer/input_layer_1/customer_embedding/to_sparse_input/values*
num_buckets *#
_output_shapes
:˙˙˙˙˙˙˙˙˙

hinput_layer/input_layer_1/customer_embedding/embedding_weights/part_0/Initializer/truncated_normal/shapeConst*X
_classN
LJloc:@input_layer/input_layer_1/customer_embedding/embedding_weights/part_0*
_output_shapes
:*
dtype0*
valueB"     

ginput_layer/input_layer_1/customer_embedding/embedding_weights/part_0/Initializer/truncated_normal/meanConst*
_output_shapes
: *
dtype0*X
_classN
LJloc:@input_layer/input_layer_1/customer_embedding/embedding_weights/part_0*
valueB
 *    

iinput_layer/input_layer_1/customer_embedding/embedding_weights/part_0/Initializer/truncated_normal/stddevConst*
_output_shapes
: *
dtype0*X
_classN
LJloc:@input_layer/input_layer_1/customer_embedding/embedding_weights/part_0*
valueB
 *  >

rinput_layer/input_layer_1/customer_embedding/embedding_weights/part_0/Initializer/truncated_normal/TruncatedNormalTruncatedNormalhinput_layer/input_layer_1/customer_embedding/embedding_weights/part_0/Initializer/truncated_normal/shape*
dtype0*
T0* 
_output_shapes
:
 *X
_classN
LJloc:@input_layer/input_layer_1/customer_embedding/embedding_weights/part_0
Ń
finput_layer/input_layer_1/customer_embedding/embedding_weights/part_0/Initializer/truncated_normal/mulMulrinput_layer/input_layer_1/customer_embedding/embedding_weights/part_0/Initializer/truncated_normal/TruncatedNormaliinput_layer/input_layer_1/customer_embedding/embedding_weights/part_0/Initializer/truncated_normal/stddev* 
_output_shapes
:
 *
T0*X
_classN
LJloc:@input_layer/input_layer_1/customer_embedding/embedding_weights/part_0
ż
binput_layer/input_layer_1/customer_embedding/embedding_weights/part_0/Initializer/truncated_normalAddfinput_layer/input_layer_1/customer_embedding/embedding_weights/part_0/Initializer/truncated_normal/mulginput_layer/input_layer_1/customer_embedding/embedding_weights/part_0/Initializer/truncated_normal/mean* 
_output_shapes
:
 *
T0*X
_classN
LJloc:@input_layer/input_layer_1/customer_embedding/embedding_weights/part_0
ó
Einput_layer/input_layer_1/customer_embedding/embedding_weights/part_0
VariableV2*X
_classN
LJloc:@input_layer/input_layer_1/customer_embedding/embedding_weights/part_0*
dtype0* 
_output_shapes
:
 *
shape:
 

Linput_layer/input_layer_1/customer_embedding/embedding_weights/part_0/AssignAssignEinput_layer/input_layer_1/customer_embedding/embedding_weights/part_0binput_layer/input_layer_1/customer_embedding/embedding_weights/part_0/Initializer/truncated_normal*X
_classN
LJloc:@input_layer/input_layer_1/customer_embedding/embedding_weights/part_0* 
_output_shapes
:
 *
T0
˘
Jinput_layer/input_layer_1/customer_embedding/embedding_weights/part_0/readIdentityEinput_layer/input_layer_1/customer_embedding/embedding_weights/part_0*X
_classN
LJloc:@input_layer/input_layer_1/customer_embedding/embedding_weights/part_0*
T0* 
_output_shapes
:
 

Sinput_layer/input_layer_1/customer_embedding/customer_embedding_weights/Slice/beginConst*
_output_shapes
:*
valueB: *
dtype0

Rinput_layer/input_layer_1/customer_embedding/customer_embedding_weights/Slice/sizeConst*
dtype0*
valueB:*
_output_shapes
:
ű
Minput_layer/input_layer_1/customer_embedding/customer_embedding_weights/SliceSliceHinput_layer/input_layer_1/customer_embedding/to_sparse_input/dense_shapeSinput_layer/input_layer_1/customer_embedding/customer_embedding_weights/Slice/beginRinput_layer/input_layer_1/customer_embedding/customer_embedding_weights/Slice/size*
_output_shapes
:*
T0	*
Index0

Minput_layer/input_layer_1/customer_embedding/customer_embedding_weights/ConstConst*
dtype0*
_output_shapes
:*
valueB: 

Linput_layer/input_layer_1/customer_embedding/customer_embedding_weights/ProdProdMinput_layer/input_layer_1/customer_embedding/customer_embedding_weights/SliceMinput_layer/input_layer_1/customer_embedding/customer_embedding_weights/Const*
_output_shapes
: *
T0	

Xinput_layer/input_layer_1/customer_embedding/customer_embedding_weights/GatherV2/indicesConst*
value	B :*
_output_shapes
: *
dtype0

Uinput_layer/input_layer_1/customer_embedding/customer_embedding_weights/GatherV2/axisConst*
_output_shapes
: *
value	B : *
dtype0

Pinput_layer/input_layer_1/customer_embedding/customer_embedding_weights/GatherV2GatherV2Hinput_layer/input_layer_1/customer_embedding/to_sparse_input/dense_shapeXinput_layer/input_layer_1/customer_embedding/customer_embedding_weights/GatherV2/indicesUinput_layer/input_layer_1/customer_embedding/customer_embedding_weights/GatherV2/axis*
_output_shapes
: *
Tparams0	*
Tindices0*
Taxis0
¤
Ninput_layer/input_layer_1/customer_embedding/customer_embedding_weights/Cast/xPackLinput_layer/input_layer_1/customer_embedding/customer_embedding_weights/ProdPinput_layer/input_layer_1/customer_embedding/customer_embedding_weights/GatherV2*
_output_shapes
:*
N*
T0	
ő
Uinput_layer/input_layer_1/customer_embedding/customer_embedding_weights/SparseReshapeSparseReshapeDinput_layer/input_layer_1/customer_embedding/to_sparse_input/indicesHinput_layer/input_layer_1/customer_embedding/to_sparse_input/dense_shapeNinput_layer/input_layer_1/customer_embedding/customer_embedding_weights/Cast/x*-
_output_shapes
:˙˙˙˙˙˙˙˙˙:
Í
^input_layer/input_layer_1/customer_embedding/customer_embedding_weights/SparseReshape/IdentityIdentity3input_layer/input_layer_1/customer_embedding/lookup*
T0	*#
_output_shapes
:˙˙˙˙˙˙˙˙˙

Vinput_layer/input_layer_1/customer_embedding/customer_embedding_weights/GreaterEqual/yConst*
value	B	 R *
_output_shapes
: *
dtype0	
Ę
Tinput_layer/input_layer_1/customer_embedding/customer_embedding_weights/GreaterEqualGreaterEqual^input_layer/input_layer_1/customer_embedding/customer_embedding_weights/SparseReshape/IdentityVinput_layer/input_layer_1/customer_embedding/customer_embedding_weights/GreaterEqual/y*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0	
Ő
Minput_layer/input_layer_1/customer_embedding/customer_embedding_weights/WhereWhereTinput_layer/input_layer_1/customer_embedding/customer_embedding_weights/GreaterEqual*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
¨
Uinput_layer/input_layer_1/customer_embedding/customer_embedding_weights/Reshape/shapeConst*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0
Ž
Oinput_layer/input_layer_1/customer_embedding/customer_embedding_weights/ReshapeReshapeMinput_layer/input_layer_1/customer_embedding/customer_embedding_weights/WhereUinput_layer/input_layer_1/customer_embedding/customer_embedding_weights/Reshape/shape*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0	

Winput_layer/input_layer_1/customer_embedding/customer_embedding_weights/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
´
Rinput_layer/input_layer_1/customer_embedding/customer_embedding_weights/GatherV2_1GatherV2Uinput_layer/input_layer_1/customer_embedding/customer_embedding_weights/SparseReshapeOinput_layer/input_layer_1/customer_embedding/customer_embedding_weights/ReshapeWinput_layer/input_layer_1/customer_embedding/customer_embedding_weights/GatherV2_1/axis*
Tindices0	*
Tparams0	*
Taxis0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

Winput_layer/input_layer_1/customer_embedding/customer_embedding_weights/GatherV2_2/axisConst*
value	B : *
_output_shapes
: *
dtype0
š
Rinput_layer/input_layer_1/customer_embedding/customer_embedding_weights/GatherV2_2GatherV2^input_layer/input_layer_1/customer_embedding/customer_embedding_weights/SparseReshape/IdentityOinput_layer/input_layer_1/customer_embedding/customer_embedding_weights/ReshapeWinput_layer/input_layer_1/customer_embedding/customer_embedding_weights/GatherV2_2/axis*
Tindices0	*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
Tparams0	*
Taxis0
Ú
Pinput_layer/input_layer_1/customer_embedding/customer_embedding_weights/IdentityIdentityWinput_layer/input_layer_1/customer_embedding/customer_embedding_weights/SparseReshape:1*
_output_shapes
:*
T0	
Ł
ainput_layer/input_layer_1/customer_embedding/customer_embedding_weights/SparseFillEmptyRows/ConstConst*
dtype0	*
value	B	 R *
_output_shapes
: 
Â
oinput_layer/input_layer_1/customer_embedding/customer_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowsSparseFillEmptyRowsRinput_layer/input_layer_1/customer_embedding/customer_embedding_weights/GatherV2_1Rinput_layer/input_layer_1/customer_embedding/customer_embedding_weights/GatherV2_2Pinput_layer/input_layer_1/customer_embedding/customer_embedding_weights/Identityainput_layer/input_layer_1/customer_embedding/customer_embedding_weights/SparseFillEmptyRows/Const*T
_output_shapesB
@:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0	
Ä
sinput_layer/input_layer_1/customer_embedding/customer_embedding_weights/embedding_lookup_sparse/strided_slice/stackConst*
valueB"        *
dtype0*
_output_shapes
:
Ć
uinput_layer/input_layer_1/customer_embedding/customer_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB"       
Ć
uinput_layer/input_layer_1/customer_embedding/customer_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2Const*
valueB"      *
_output_shapes
:*
dtype0
Ć
minput_layer/input_layer_1/customer_embedding/customer_embedding_weights/embedding_lookup_sparse/strided_sliceStridedSliceoinput_layer/input_layer_1/customer_embedding/customer_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowssinput_layer/input_layer_1/customer_embedding/customer_embedding_weights/embedding_lookup_sparse/strided_slice/stackuinput_layer/input_layer_1/customer_embedding/customer_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1uinput_layer/input_layer_1/customer_embedding/customer_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2*
T0	*

begin_mask*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
end_mask*
shrink_axis_mask*
Index0

dinput_layer/input_layer_1/customer_embedding/customer_embedding_weights/embedding_lookup_sparse/CastCastminput_layer/input_layer_1/customer_embedding/customer_embedding_weights/embedding_lookup_sparse/strided_slice*

DstT0*

SrcT0	*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
finput_layer/input_layer_1/customer_embedding/customer_embedding_weights/embedding_lookup_sparse/UniqueUniqueqinput_layer/input_layer_1/customer_embedding/customer_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:1*
T0	*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙

uinput_layer/input_layer_1/customer_embedding/customer_embedding_weights/embedding_lookup_sparse/embedding_lookup/axisConst*
value	B : *X
_classN
LJloc:@input_layer/input_layer_1/customer_embedding/embedding_weights/part_0*
dtype0*
_output_shapes
: 
Ö
pinput_layer/input_layer_1/customer_embedding/customer_embedding_weights/embedding_lookup_sparse/embedding_lookupGatherV2Jinput_layer/input_layer_1/customer_embedding/embedding_weights/part_0/readfinput_layer/input_layer_1/customer_embedding/customer_embedding_weights/embedding_lookup_sparse/Uniqueuinput_layer/input_layer_1/customer_embedding/customer_embedding_weights/embedding_lookup_sparse/embedding_lookup/axis*
Tparams0*
Taxis0*X
_classN
LJloc:@input_layer/input_layer_1/customer_embedding/embedding_weights/part_0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
Tindices0	
Š
yinput_layer/input_layer_1/customer_embedding/customer_embedding_weights/embedding_lookup_sparse/embedding_lookup/IdentityIdentitypinput_layer/input_layer_1/customer_embedding/customer_embedding_weights/embedding_lookup_sparse/embedding_lookup*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
ń
_input_layer/input_layer_1/customer_embedding/customer_embedding_weights/embedding_lookup_sparseSparseSegmentMeanyinput_layer/input_layer_1/customer_embedding/customer_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identityhinput_layer/input_layer_1/customer_embedding/customer_embedding_weights/embedding_lookup_sparse/Unique:1dinput_layer/input_layer_1/customer_embedding/customer_embedding_weights/embedding_lookup_sparse/Cast*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
¨
Winput_layer/input_layer_1/customer_embedding/customer_embedding_weights/Reshape_1/shapeConst*
valueB"˙˙˙˙   *
_output_shapes
:*
dtype0
Ú
Qinput_layer/input_layer_1/customer_embedding/customer_embedding_weights/Reshape_1Reshapeqinput_layer/input_layer_1/customer_embedding/customer_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:2Winput_layer/input_layer_1/customer_embedding/customer_embedding_weights/Reshape_1/shape*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

Ü
Minput_layer/input_layer_1/customer_embedding/customer_embedding_weights/ShapeShape_input_layer/input_layer_1/customer_embedding/customer_embedding_weights/embedding_lookup_sparse*
_output_shapes
:*
T0
Ľ
[input_layer/input_layer_1/customer_embedding/customer_embedding_weights/strided_slice/stackConst*
valueB:*
dtype0*
_output_shapes
:
§
]input_layer/input_layer_1/customer_embedding/customer_embedding_weights/strided_slice/stack_1Const*
dtype0*
valueB:*
_output_shapes
:
§
]input_layer/input_layer_1/customer_embedding/customer_embedding_weights/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:

Uinput_layer/input_layer_1/customer_embedding/customer_embedding_weights/strided_sliceStridedSliceMinput_layer/input_layer_1/customer_embedding/customer_embedding_weights/Shape[input_layer/input_layer_1/customer_embedding/customer_embedding_weights/strided_slice/stack]input_layer/input_layer_1/customer_embedding/customer_embedding_weights/strided_slice/stack_1]input_layer/input_layer_1/customer_embedding/customer_embedding_weights/strided_slice/stack_2*
T0*
_output_shapes
: *
Index0*
shrink_axis_mask

Oinput_layer/input_layer_1/customer_embedding/customer_embedding_weights/stack/0Const*
value	B :*
_output_shapes
: *
dtype0
Ť
Minput_layer/input_layer_1/customer_embedding/customer_embedding_weights/stackPackOinput_layer/input_layer_1/customer_embedding/customer_embedding_weights/stack/0Uinput_layer/input_layer_1/customer_embedding/customer_embedding_weights/strided_slice*
_output_shapes
:*
N*
T0
ą
Linput_layer/input_layer_1/customer_embedding/customer_embedding_weights/TileTileQinput_layer/input_layer_1/customer_embedding/customer_embedding_weights/Reshape_1Minput_layer/input_layer_1/customer_embedding/customer_embedding_weights/stack*
T0
*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
ň
Rinput_layer/input_layer_1/customer_embedding/customer_embedding_weights/zeros_like	ZerosLike_input_layer/input_layer_1/customer_embedding/customer_embedding_weights/embedding_lookup_sparse*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

Ginput_layer/input_layer_1/customer_embedding/customer_embedding_weightsSelectLinput_layer/input_layer_1/customer_embedding/customer_embedding_weights/TileRinput_layer/input_layer_1/customer_embedding/customer_embedding_weights/zeros_like_input_layer/input_layer_1/customer_embedding/customer_embedding_weights/embedding_lookup_sparse*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ô
Ninput_layer/input_layer_1/customer_embedding/customer_embedding_weights/Cast_1CastHinput_layer/input_layer_1/customer_embedding/to_sparse_input/dense_shape*
_output_shapes
:*

DstT0*

SrcT0	

Uinput_layer/input_layer_1/customer_embedding/customer_embedding_weights/Slice_1/beginConst*
dtype0*
valueB: *
_output_shapes
:

Tinput_layer/input_layer_1/customer_embedding/customer_embedding_weights/Slice_1/sizeConst*
valueB:*
dtype0*
_output_shapes
:

Oinput_layer/input_layer_1/customer_embedding/customer_embedding_weights/Slice_1SliceNinput_layer/input_layer_1/customer_embedding/customer_embedding_weights/Cast_1Uinput_layer/input_layer_1/customer_embedding/customer_embedding_weights/Slice_1/beginTinput_layer/input_layer_1/customer_embedding/customer_embedding_weights/Slice_1/size*
_output_shapes
:*
T0*
Index0
Ć
Oinput_layer/input_layer_1/customer_embedding/customer_embedding_weights/Shape_1ShapeGinput_layer/input_layer_1/customer_embedding/customer_embedding_weights*
T0*
_output_shapes
:

Uinput_layer/input_layer_1/customer_embedding/customer_embedding_weights/Slice_2/beginConst*
_output_shapes
:*
valueB:*
dtype0
§
Tinput_layer/input_layer_1/customer_embedding/customer_embedding_weights/Slice_2/sizeConst*
dtype0*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙

Oinput_layer/input_layer_1/customer_embedding/customer_embedding_weights/Slice_2SliceOinput_layer/input_layer_1/customer_embedding/customer_embedding_weights/Shape_1Uinput_layer/input_layer_1/customer_embedding/customer_embedding_weights/Slice_2/beginTinput_layer/input_layer_1/customer_embedding/customer_embedding_weights/Slice_2/size*
T0*
_output_shapes
:*
Index0

Sinput_layer/input_layer_1/customer_embedding/customer_embedding_weights/concat/axisConst*
dtype0*
_output_shapes
: *
value	B : 
˙
Ninput_layer/input_layer_1/customer_embedding/customer_embedding_weights/concatConcatV2Oinput_layer/input_layer_1/customer_embedding/customer_embedding_weights/Slice_1Oinput_layer/input_layer_1/customer_embedding/customer_embedding_weights/Slice_2Sinput_layer/input_layer_1/customer_embedding/customer_embedding_weights/concat/axis*
N*
T0*
_output_shapes
:
§
Qinput_layer/input_layer_1/customer_embedding/customer_embedding_weights/Reshape_2ReshapeGinput_layer/input_layer_1/customer_embedding/customer_embedding_weightsNinput_layer/input_layer_1/customer_embedding/customer_embedding_weights/concat*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
ł
2input_layer/input_layer_1/customer_embedding/ShapeShapeQinput_layer/input_layer_1/customer_embedding/customer_embedding_weights/Reshape_2*
_output_shapes
:*
T0

@input_layer/input_layer_1/customer_embedding/strided_slice/stackConst*
valueB: *
_output_shapes
:*
dtype0

Binput_layer/input_layer_1/customer_embedding/strided_slice/stack_1Const*
valueB:*
_output_shapes
:*
dtype0

Binput_layer/input_layer_1/customer_embedding/strided_slice/stack_2Const*
dtype0*
valueB:*
_output_shapes
:

:input_layer/input_layer_1/customer_embedding/strided_sliceStridedSlice2input_layer/input_layer_1/customer_embedding/Shape@input_layer/input_layer_1/customer_embedding/strided_slice/stackBinput_layer/input_layer_1/customer_embedding/strided_slice/stack_1Binput_layer/input_layer_1/customer_embedding/strided_slice/stack_2*
Index0*
shrink_axis_mask*
_output_shapes
: *
T0
~
<input_layer/input_layer_1/customer_embedding/Reshape/shape/1Const*
value	B :*
dtype0*
_output_shapes
: 
ę
:input_layer/input_layer_1/customer_embedding/Reshape/shapePack:input_layer/input_layer_1/customer_embedding/strided_slice<input_layer/input_layer_1/customer_embedding/Reshape/shape/1*
N*
_output_shapes
:*
T0

4input_layer/input_layer_1/customer_embedding/ReshapeReshapeQinput_layer/input_layer_1/customer_embedding/customer_embedding_weights/Reshape_2:input_layer/input_layer_1/customer_embedding/Reshape/shape*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

Dinput_layer/input_layer_1/final_gender_code_embedding/ExpandDims/dimConst*
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙*
_output_shapes
: 
Ö
@input_layer/input_layer_1/final_gender_code_embedding/ExpandDims
ExpandDimsPlaceholder_11Dinput_layer/input_layer_1/final_gender_code_embedding/ExpandDims/dim*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

Tinput_layer/input_layer_1/final_gender_code_embedding/to_sparse_input/ignore_value/xConst*
valueB B *
dtype0*
_output_shapes
: 
¤
Ninput_layer/input_layer_1/final_gender_code_embedding/to_sparse_input/NotEqualNotEqual@input_layer/input_layer_1/final_gender_code_embedding/ExpandDimsTinput_layer/input_layer_1/final_gender_code_embedding/to_sparse_input/ignore_value/x*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ď
Minput_layer/input_layer_1/final_gender_code_embedding/to_sparse_input/indicesWhereNinput_layer/input_layer_1/final_gender_code_embedding/to_sparse_input/NotEqual*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
­
Linput_layer/input_layer_1/final_gender_code_embedding/to_sparse_input/valuesGatherNd@input_layer/input_layer_1/final_gender_code_embedding/ExpandDimsMinput_layer/input_layer_1/final_gender_code_embedding/to_sparse_input/indices*
Tparams0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
Tindices0	
Ń
Qinput_layer/input_layer_1/final_gender_code_embedding/to_sparse_input/dense_shapeShape@input_layer/input_layer_1/final_gender_code_embedding/ExpandDims*
T0*
_output_shapes
:*
out_type0	
Ü
<input_layer/input_layer_1/final_gender_code_embedding/lookupStringToHashBucketFastLinput_layer/input_layer_1/final_gender_code_embedding/to_sparse_input/values*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
num_buckets

Ľ
qinput_layer/input_layer_1/final_gender_code_embedding/embedding_weights/part_0/Initializer/truncated_normal/shapeConst*
_output_shapes
:*
valueB"
      *a
_classW
USloc:@input_layer/input_layer_1/final_gender_code_embedding/embedding_weights/part_0*
dtype0

pinput_layer/input_layer_1/final_gender_code_embedding/embedding_weights/part_0/Initializer/truncated_normal/meanConst*a
_classW
USloc:@input_layer/input_layer_1/final_gender_code_embedding/embedding_weights/part_0*
valueB
 *    *
dtype0*
_output_shapes
: 

rinput_layer/input_layer_1/final_gender_code_embedding/embedding_weights/part_0/Initializer/truncated_normal/stddevConst*
dtype0*
valueB
 *  >*
_output_shapes
: *a
_classW
USloc:@input_layer/input_layer_1/final_gender_code_embedding/embedding_weights/part_0

{input_layer/input_layer_1/final_gender_code_embedding/embedding_weights/part_0/Initializer/truncated_normal/TruncatedNormalTruncatedNormalqinput_layer/input_layer_1/final_gender_code_embedding/embedding_weights/part_0/Initializer/truncated_normal/shape*
T0*
dtype0*a
_classW
USloc:@input_layer/input_layer_1/final_gender_code_embedding/embedding_weights/part_0*
_output_shapes

:

ó
oinput_layer/input_layer_1/final_gender_code_embedding/embedding_weights/part_0/Initializer/truncated_normal/mulMul{input_layer/input_layer_1/final_gender_code_embedding/embedding_weights/part_0/Initializer/truncated_normal/TruncatedNormalrinput_layer/input_layer_1/final_gender_code_embedding/embedding_weights/part_0/Initializer/truncated_normal/stddev*a
_classW
USloc:@input_layer/input_layer_1/final_gender_code_embedding/embedding_weights/part_0*
_output_shapes

:
*
T0
á
kinput_layer/input_layer_1/final_gender_code_embedding/embedding_weights/part_0/Initializer/truncated_normalAddoinput_layer/input_layer_1/final_gender_code_embedding/embedding_weights/part_0/Initializer/truncated_normal/mulpinput_layer/input_layer_1/final_gender_code_embedding/embedding_weights/part_0/Initializer/truncated_normal/mean*a
_classW
USloc:@input_layer/input_layer_1/final_gender_code_embedding/embedding_weights/part_0*
_output_shapes

:
*
T0

Ninput_layer/input_layer_1/final_gender_code_embedding/embedding_weights/part_0
VariableV2*
_output_shapes

:
*a
_classW
USloc:@input_layer/input_layer_1/final_gender_code_embedding/embedding_weights/part_0*
shape
:
*
dtype0
¨
Uinput_layer/input_layer_1/final_gender_code_embedding/embedding_weights/part_0/AssignAssignNinput_layer/input_layer_1/final_gender_code_embedding/embedding_weights/part_0kinput_layer/input_layer_1/final_gender_code_embedding/embedding_weights/part_0/Initializer/truncated_normal*
T0*a
_classW
USloc:@input_layer/input_layer_1/final_gender_code_embedding/embedding_weights/part_0*
_output_shapes

:

ť
Sinput_layer/input_layer_1/final_gender_code_embedding/embedding_weights/part_0/readIdentityNinput_layer/input_layer_1/final_gender_code_embedding/embedding_weights/part_0*
_output_shapes

:
*a
_classW
USloc:@input_layer/input_layer_1/final_gender_code_embedding/embedding_weights/part_0*
T0
Ż
einput_layer/input_layer_1/final_gender_code_embedding/final_gender_code_embedding_weights/Slice/beginConst*
dtype0*
valueB: *
_output_shapes
:
Ž
dinput_layer/input_layer_1/final_gender_code_embedding/final_gender_code_embedding_weights/Slice/sizeConst*
dtype0*
valueB:*
_output_shapes
:
ş
_input_layer/input_layer_1/final_gender_code_embedding/final_gender_code_embedding_weights/SliceSliceQinput_layer/input_layer_1/final_gender_code_embedding/to_sparse_input/dense_shapeeinput_layer/input_layer_1/final_gender_code_embedding/final_gender_code_embedding_weights/Slice/begindinput_layer/input_layer_1/final_gender_code_embedding/final_gender_code_embedding_weights/Slice/size*
_output_shapes
:*
Index0*
T0	
Š
_input_layer/input_layer_1/final_gender_code_embedding/final_gender_code_embedding_weights/ConstConst*
dtype0*
valueB: *
_output_shapes
:
É
^input_layer/input_layer_1/final_gender_code_embedding/final_gender_code_embedding_weights/ProdProd_input_layer/input_layer_1/final_gender_code_embedding/final_gender_code_embedding_weights/Slice_input_layer/input_layer_1/final_gender_code_embedding/final_gender_code_embedding_weights/Const*
T0	*
_output_shapes
: 
Ź
jinput_layer/input_layer_1/final_gender_code_embedding/final_gender_code_embedding_weights/GatherV2/indicesConst*
dtype0*
_output_shapes
: *
value	B :
Š
ginput_layer/input_layer_1/final_gender_code_embedding/final_gender_code_embedding_weights/GatherV2/axisConst*
_output_shapes
: *
value	B : *
dtype0
Ú
binput_layer/input_layer_1/final_gender_code_embedding/final_gender_code_embedding_weights/GatherV2GatherV2Qinput_layer/input_layer_1/final_gender_code_embedding/to_sparse_input/dense_shapejinput_layer/input_layer_1/final_gender_code_embedding/final_gender_code_embedding_weights/GatherV2/indicesginput_layer/input_layer_1/final_gender_code_embedding/final_gender_code_embedding_weights/GatherV2/axis*
Tindices0*
Taxis0*
_output_shapes
: *
Tparams0	
Ú
`input_layer/input_layer_1/final_gender_code_embedding/final_gender_code_embedding_weights/Cast/xPack^input_layer/input_layer_1/final_gender_code_embedding/final_gender_code_embedding_weights/Prodbinput_layer/input_layer_1/final_gender_code_embedding/final_gender_code_embedding_weights/GatherV2*
T0	*
N*
_output_shapes
:
Ť
ginput_layer/input_layer_1/final_gender_code_embedding/final_gender_code_embedding_weights/SparseReshapeSparseReshapeMinput_layer/input_layer_1/final_gender_code_embedding/to_sparse_input/indicesQinput_layer/input_layer_1/final_gender_code_embedding/to_sparse_input/dense_shape`input_layer/input_layer_1/final_gender_code_embedding/final_gender_code_embedding_weights/Cast/x*-
_output_shapes
:˙˙˙˙˙˙˙˙˙:
č
pinput_layer/input_layer_1/final_gender_code_embedding/final_gender_code_embedding_weights/SparseReshape/IdentityIdentity<input_layer/input_layer_1/final_gender_code_embedding/lookup*
T0	*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ş
hinput_layer/input_layer_1/final_gender_code_embedding/final_gender_code_embedding_weights/GreaterEqual/yConst*
_output_shapes
: *
dtype0	*
value	B	 R 

finput_layer/input_layer_1/final_gender_code_embedding/final_gender_code_embedding_weights/GreaterEqualGreaterEqualpinput_layer/input_layer_1/final_gender_code_embedding/final_gender_code_embedding_weights/SparseReshape/Identityhinput_layer/input_layer_1/final_gender_code_embedding/final_gender_code_embedding_weights/GreaterEqual/y*
T0	*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
ů
_input_layer/input_layer_1/final_gender_code_embedding/final_gender_code_embedding_weights/WhereWherefinput_layer/input_layer_1/final_gender_code_embedding/final_gender_code_embedding_weights/GreaterEqual*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
ş
ginput_layer/input_layer_1/final_gender_code_embedding/final_gender_code_embedding_weights/Reshape/shapeConst*
dtype0*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙
ä
ainput_layer/input_layer_1/final_gender_code_embedding/final_gender_code_embedding_weights/ReshapeReshape_input_layer/input_layer_1/final_gender_code_embedding/final_gender_code_embedding_weights/Whereginput_layer/input_layer_1/final_gender_code_embedding/final_gender_code_embedding_weights/Reshape/shape*
T0	*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ť
iinput_layer/input_layer_1/final_gender_code_embedding/final_gender_code_embedding_weights/GatherV2_1/axisConst*
value	B : *
_output_shapes
: *
dtype0
ü
dinput_layer/input_layer_1/final_gender_code_embedding/final_gender_code_embedding_weights/GatherV2_1GatherV2ginput_layer/input_layer_1/final_gender_code_embedding/final_gender_code_embedding_weights/SparseReshapeainput_layer/input_layer_1/final_gender_code_embedding/final_gender_code_embedding_weights/Reshapeiinput_layer/input_layer_1/final_gender_code_embedding/final_gender_code_embedding_weights/GatherV2_1/axis*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
Tindices0	*
Taxis0*
Tparams0	
Ť
iinput_layer/input_layer_1/final_gender_code_embedding/final_gender_code_embedding_weights/GatherV2_2/axisConst*
value	B : *
_output_shapes
: *
dtype0

dinput_layer/input_layer_1/final_gender_code_embedding/final_gender_code_embedding_weights/GatherV2_2GatherV2pinput_layer/input_layer_1/final_gender_code_embedding/final_gender_code_embedding_weights/SparseReshape/Identityainput_layer/input_layer_1/final_gender_code_embedding/final_gender_code_embedding_weights/Reshapeiinput_layer/input_layer_1/final_gender_code_embedding/final_gender_code_embedding_weights/GatherV2_2/axis*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
Tparams0	*
Taxis0*
Tindices0	
ţ
binput_layer/input_layer_1/final_gender_code_embedding/final_gender_code_embedding_weights/IdentityIdentityiinput_layer/input_layer_1/final_gender_code_embedding/final_gender_code_embedding_weights/SparseReshape:1*
_output_shapes
:*
T0	
ľ
sinput_layer/input_layer_1/final_gender_code_embedding/final_gender_code_embedding_weights/SparseFillEmptyRows/ConstConst*
value	B	 R *
dtype0	*
_output_shapes
: 

input_layer/input_layer_1/final_gender_code_embedding/final_gender_code_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowsSparseFillEmptyRowsdinput_layer/input_layer_1/final_gender_code_embedding/final_gender_code_embedding_weights/GatherV2_1dinput_layer/input_layer_1/final_gender_code_embedding/final_gender_code_embedding_weights/GatherV2_2binput_layer/input_layer_1/final_gender_code_embedding/final_gender_code_embedding_weights/Identitysinput_layer/input_layer_1/final_gender_code_embedding/final_gender_code_embedding_weights/SparseFillEmptyRows/Const*T
_output_shapesB
@:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0	
×
input_layer/input_layer_1/final_gender_code_embedding/final_gender_code_embedding_weights/embedding_lookup_sparse/strided_slice/stackConst*
valueB"        *
dtype0*
_output_shapes
:
Ů
input_layer/input_layer_1/final_gender_code_embedding/final_gender_code_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1Const*
_output_shapes
:*
valueB"       *
dtype0
Ů
input_layer/input_layer_1/final_gender_code_embedding/final_gender_code_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:
¤
input_layer/input_layer_1/final_gender_code_embedding/final_gender_code_embedding_weights/embedding_lookup_sparse/strided_sliceStridedSliceinput_layer/input_layer_1/final_gender_code_embedding/final_gender_code_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowsinput_layer/input_layer_1/final_gender_code_embedding/final_gender_code_embedding_weights/embedding_lookup_sparse/strided_slice/stackinput_layer/input_layer_1/final_gender_code_embedding/final_gender_code_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1input_layer/input_layer_1/final_gender_code_embedding/final_gender_code_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
end_mask*
shrink_axis_mask*
T0	*

begin_mask*
Index0
ź
vinput_layer/input_layer_1/final_gender_code_embedding/final_gender_code_embedding_weights/embedding_lookup_sparse/CastCastinput_layer/input_layer_1/final_gender_code_embedding/final_gender_code_embedding_weights/embedding_lookup_sparse/strided_slice*

SrcT0	*

DstT0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ĺ
xinput_layer/input_layer_1/final_gender_code_embedding/final_gender_code_embedding_weights/embedding_lookup_sparse/UniqueUniqueinput_layer/input_layer_1/final_gender_code_embedding/final_gender_code_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0	
­
input_layer/input_layer_1/final_gender_code_embedding/final_gender_code_embedding_weights/embedding_lookup_sparse/embedding_lookup/axisConst*
value	B : *
_output_shapes
: *a
_classW
USloc:@input_layer/input_layer_1/final_gender_code_embedding/embedding_weights/part_0*
dtype0
 
input_layer/input_layer_1/final_gender_code_embedding/final_gender_code_embedding_weights/embedding_lookup_sparse/embedding_lookupGatherV2Sinput_layer/input_layer_1/final_gender_code_embedding/embedding_weights/part_0/readxinput_layer/input_layer_1/final_gender_code_embedding/final_gender_code_embedding_weights/embedding_lookup_sparse/Uniqueinput_layer/input_layer_1/final_gender_code_embedding/final_gender_code_embedding_weights/embedding_lookup_sparse/embedding_lookup/axis*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
Tindices0	*a
_classW
USloc:@input_layer/input_layer_1/final_gender_code_embedding/embedding_weights/part_0*
Taxis0*
Tparams0
Ď
input_layer/input_layer_1/final_gender_code_embedding/final_gender_code_embedding_weights/embedding_lookup_sparse/embedding_lookup/IdentityIdentityinput_layer/input_layer_1/final_gender_code_embedding/final_gender_code_embedding_weights/embedding_lookup_sparse/embedding_lookup*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
ş
qinput_layer/input_layer_1/final_gender_code_embedding/final_gender_code_embedding_weights/embedding_lookup_sparseSparseSegmentMeaninput_layer/input_layer_1/final_gender_code_embedding/final_gender_code_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identityzinput_layer/input_layer_1/final_gender_code_embedding/final_gender_code_embedding_weights/embedding_lookup_sparse/Unique:1vinput_layer/input_layer_1/final_gender_code_embedding/final_gender_code_embedding_weights/embedding_lookup_sparse/Cast*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
ş
iinput_layer/input_layer_1/final_gender_code_embedding/final_gender_code_embedding_weights/Reshape_1/shapeConst*
valueB"˙˙˙˙   *
_output_shapes
:*
dtype0

cinput_layer/input_layer_1/final_gender_code_embedding/final_gender_code_embedding_weights/Reshape_1Reshapeinput_layer/input_layer_1/final_gender_code_embedding/final_gender_code_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:2iinput_layer/input_layer_1/final_gender_code_embedding/final_gender_code_embedding_weights/Reshape_1/shape*
T0
*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

_input_layer/input_layer_1/final_gender_code_embedding/final_gender_code_embedding_weights/ShapeShapeqinput_layer/input_layer_1/final_gender_code_embedding/final_gender_code_embedding_weights/embedding_lookup_sparse*
_output_shapes
:*
T0
ˇ
minput_layer/input_layer_1/final_gender_code_embedding/final_gender_code_embedding_weights/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB:
š
oinput_layer/input_layer_1/final_gender_code_embedding/final_gender_code_embedding_weights/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
š
oinput_layer/input_layer_1/final_gender_code_embedding/final_gender_code_embedding_weights/strided_slice/stack_2Const*
_output_shapes
:*
valueB:*
dtype0
ď
ginput_layer/input_layer_1/final_gender_code_embedding/final_gender_code_embedding_weights/strided_sliceStridedSlice_input_layer/input_layer_1/final_gender_code_embedding/final_gender_code_embedding_weights/Shapeminput_layer/input_layer_1/final_gender_code_embedding/final_gender_code_embedding_weights/strided_slice/stackoinput_layer/input_layer_1/final_gender_code_embedding/final_gender_code_embedding_weights/strided_slice/stack_1oinput_layer/input_layer_1/final_gender_code_embedding/final_gender_code_embedding_weights/strided_slice/stack_2*
T0*
Index0*
_output_shapes
: *
shrink_axis_mask
Ł
ainput_layer/input_layer_1/final_gender_code_embedding/final_gender_code_embedding_weights/stack/0Const*
dtype0*
_output_shapes
: *
value	B :
á
_input_layer/input_layer_1/final_gender_code_embedding/final_gender_code_embedding_weights/stackPackainput_layer/input_layer_1/final_gender_code_embedding/final_gender_code_embedding_weights/stack/0ginput_layer/input_layer_1/final_gender_code_embedding/final_gender_code_embedding_weights/strided_slice*
N*
_output_shapes
:*
T0
ç
^input_layer/input_layer_1/final_gender_code_embedding/final_gender_code_embedding_weights/TileTilecinput_layer/input_layer_1/final_gender_code_embedding/final_gender_code_embedding_weights/Reshape_1_input_layer/input_layer_1/final_gender_code_embedding/final_gender_code_embedding_weights/stack*
T0
*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙

dinput_layer/input_layer_1/final_gender_code_embedding/final_gender_code_embedding_weights/zeros_like	ZerosLikeqinput_layer/input_layer_1/final_gender_code_embedding/final_gender_code_embedding_weights/embedding_lookup_sparse*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Î
Yinput_layer/input_layer_1/final_gender_code_embedding/final_gender_code_embedding_weightsSelect^input_layer/input_layer_1/final_gender_code_embedding/final_gender_code_embedding_weights/Tiledinput_layer/input_layer_1/final_gender_code_embedding/final_gender_code_embedding_weights/zeros_likeqinput_layer/input_layer_1/final_gender_code_embedding/final_gender_code_embedding_weights/embedding_lookup_sparse*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
ď
`input_layer/input_layer_1/final_gender_code_embedding/final_gender_code_embedding_weights/Cast_1CastQinput_layer/input_layer_1/final_gender_code_embedding/to_sparse_input/dense_shape*

DstT0*
_output_shapes
:*

SrcT0	
ą
ginput_layer/input_layer_1/final_gender_code_embedding/final_gender_code_embedding_weights/Slice_1/beginConst*
_output_shapes
:*
valueB: *
dtype0
°
finput_layer/input_layer_1/final_gender_code_embedding/final_gender_code_embedding_weights/Slice_1/sizeConst*
_output_shapes
:*
dtype0*
valueB:
Ď
ainput_layer/input_layer_1/final_gender_code_embedding/final_gender_code_embedding_weights/Slice_1Slice`input_layer/input_layer_1/final_gender_code_embedding/final_gender_code_embedding_weights/Cast_1ginput_layer/input_layer_1/final_gender_code_embedding/final_gender_code_embedding_weights/Slice_1/beginfinput_layer/input_layer_1/final_gender_code_embedding/final_gender_code_embedding_weights/Slice_1/size*
_output_shapes
:*
T0*
Index0
ę
ainput_layer/input_layer_1/final_gender_code_embedding/final_gender_code_embedding_weights/Shape_1ShapeYinput_layer/input_layer_1/final_gender_code_embedding/final_gender_code_embedding_weights*
T0*
_output_shapes
:
ą
ginput_layer/input_layer_1/final_gender_code_embedding/final_gender_code_embedding_weights/Slice_2/beginConst*
_output_shapes
:*
valueB:*
dtype0
š
finput_layer/input_layer_1/final_gender_code_embedding/final_gender_code_embedding_weights/Slice_2/sizeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
_output_shapes
:*
dtype0
Đ
ainput_layer/input_layer_1/final_gender_code_embedding/final_gender_code_embedding_weights/Slice_2Sliceainput_layer/input_layer_1/final_gender_code_embedding/final_gender_code_embedding_weights/Shape_1ginput_layer/input_layer_1/final_gender_code_embedding/final_gender_code_embedding_weights/Slice_2/beginfinput_layer/input_layer_1/final_gender_code_embedding/final_gender_code_embedding_weights/Slice_2/size*
T0*
_output_shapes
:*
Index0
§
einput_layer/input_layer_1/final_gender_code_embedding/final_gender_code_embedding_weights/concat/axisConst*
_output_shapes
: *
value	B : *
dtype0
Ç
`input_layer/input_layer_1/final_gender_code_embedding/final_gender_code_embedding_weights/concatConcatV2ainput_layer/input_layer_1/final_gender_code_embedding/final_gender_code_embedding_weights/Slice_1ainput_layer/input_layer_1/final_gender_code_embedding/final_gender_code_embedding_weights/Slice_2einput_layer/input_layer_1/final_gender_code_embedding/final_gender_code_embedding_weights/concat/axis*
T0*
N*
_output_shapes
:
Ý
cinput_layer/input_layer_1/final_gender_code_embedding/final_gender_code_embedding_weights/Reshape_2ReshapeYinput_layer/input_layer_1/final_gender_code_embedding/final_gender_code_embedding_weights`input_layer/input_layer_1/final_gender_code_embedding/final_gender_code_embedding_weights/concat*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Î
;input_layer/input_layer_1/final_gender_code_embedding/ShapeShapecinput_layer/input_layer_1/final_gender_code_embedding/final_gender_code_embedding_weights/Reshape_2*
_output_shapes
:*
T0

Iinput_layer/input_layer_1/final_gender_code_embedding/strided_slice/stackConst*
valueB: *
_output_shapes
:*
dtype0

Kinput_layer/input_layer_1/final_gender_code_embedding/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:

Kinput_layer/input_layer_1/final_gender_code_embedding/strided_slice/stack_2Const*
valueB:*
_output_shapes
:*
dtype0
ť
Cinput_layer/input_layer_1/final_gender_code_embedding/strided_sliceStridedSlice;input_layer/input_layer_1/final_gender_code_embedding/ShapeIinput_layer/input_layer_1/final_gender_code_embedding/strided_slice/stackKinput_layer/input_layer_1/final_gender_code_embedding/strided_slice/stack_1Kinput_layer/input_layer_1/final_gender_code_embedding/strided_slice/stack_2*
T0*
shrink_axis_mask*
Index0*
_output_shapes
: 

Einput_layer/input_layer_1/final_gender_code_embedding/Reshape/shape/1Const*
value	B :*
_output_shapes
: *
dtype0

Cinput_layer/input_layer_1/final_gender_code_embedding/Reshape/shapePackCinput_layer/input_layer_1/final_gender_code_embedding/strided_sliceEinput_layer/input_layer_1/final_gender_code_embedding/Reshape/shape/1*
T0*
_output_shapes
:*
N
¤
=input_layer/input_layer_1/final_gender_code_embedding/ReshapeReshapecinput_layer/input_layer_1/final_gender_code_embedding/final_gender_code_embedding_weights/Reshape_2Cinput_layer/input_layer_1/final_gender_code_embedding/Reshape/shape*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

6input_layer/input_layer_1/pid_embedding/ExpandDims/dimConst*
_output_shapes
: *
valueB :
˙˙˙˙˙˙˙˙˙*
dtype0
š
2input_layer/input_layer_1/pid_embedding/ExpandDims
ExpandDimsPlaceholder_26input_layer/input_layer_1/pid_embedding/ExpandDims/dim*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

Finput_layer/input_layer_1/pid_embedding/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
valueB B *
dtype0
ú
@input_layer/input_layer_1/pid_embedding/to_sparse_input/NotEqualNotEqual2input_layer/input_layer_1/pid_embedding/ExpandDimsFinput_layer/input_layer_1/pid_embedding/to_sparse_input/ignore_value/x*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
ł
?input_layer/input_layer_1/pid_embedding/to_sparse_input/indicesWhere@input_layer/input_layer_1/pid_embedding/to_sparse_input/NotEqual*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

>input_layer/input_layer_1/pid_embedding/to_sparse_input/valuesGatherNd2input_layer/input_layer_1/pid_embedding/ExpandDims?input_layer/input_layer_1/pid_embedding/to_sparse_input/indices*
Tindices0	*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
Tparams0
ľ
Cinput_layer/input_layer_1/pid_embedding/to_sparse_input/dense_shapeShape2input_layer/input_layer_1/pid_embedding/ExpandDims*
_output_shapes
:*
T0*
out_type0	
Ŕ
.input_layer/input_layer_1/pid_embedding/lookupStringToHashBucketFast>input_layer/input_layer_1/pid_embedding/to_sparse_input/values*
num_buckets
*#
_output_shapes
:˙˙˙˙˙˙˙˙˙

cinput_layer/input_layer_1/pid_embedding/embedding_weights/part_0/Initializer/truncated_normal/shapeConst*
dtype0*
_output_shapes
:*S
_classI
GEloc:@input_layer/input_layer_1/pid_embedding/embedding_weights/part_0*
valueB"
      
ü
binput_layer/input_layer_1/pid_embedding/embedding_weights/part_0/Initializer/truncated_normal/meanConst*
_output_shapes
: *
valueB
 *    *
dtype0*S
_classI
GEloc:@input_layer/input_layer_1/pid_embedding/embedding_weights/part_0
ţ
dinput_layer/input_layer_1/pid_embedding/embedding_weights/part_0/Initializer/truncated_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  >*S
_classI
GEloc:@input_layer/input_layer_1/pid_embedding/embedding_weights/part_0
đ
minput_layer/input_layer_1/pid_embedding/embedding_weights/part_0/Initializer/truncated_normal/TruncatedNormalTruncatedNormalcinput_layer/input_layer_1/pid_embedding/embedding_weights/part_0/Initializer/truncated_normal/shape*S
_classI
GEloc:@input_layer/input_layer_1/pid_embedding/embedding_weights/part_0*
_output_shapes

:
*
T0*
dtype0
ť
ainput_layer/input_layer_1/pid_embedding/embedding_weights/part_0/Initializer/truncated_normal/mulMulminput_layer/input_layer_1/pid_embedding/embedding_weights/part_0/Initializer/truncated_normal/TruncatedNormaldinput_layer/input_layer_1/pid_embedding/embedding_weights/part_0/Initializer/truncated_normal/stddev*S
_classI
GEloc:@input_layer/input_layer_1/pid_embedding/embedding_weights/part_0*
T0*
_output_shapes

:

Š
]input_layer/input_layer_1/pid_embedding/embedding_weights/part_0/Initializer/truncated_normalAddainput_layer/input_layer_1/pid_embedding/embedding_weights/part_0/Initializer/truncated_normal/mulbinput_layer/input_layer_1/pid_embedding/embedding_weights/part_0/Initializer/truncated_normal/mean*S
_classI
GEloc:@input_layer/input_layer_1/pid_embedding/embedding_weights/part_0*
T0*
_output_shapes

:

ĺ
@input_layer/input_layer_1/pid_embedding/embedding_weights/part_0
VariableV2*
shape
:
*
_output_shapes

:
*
dtype0*S
_classI
GEloc:@input_layer/input_layer_1/pid_embedding/embedding_weights/part_0
đ
Ginput_layer/input_layer_1/pid_embedding/embedding_weights/part_0/AssignAssign@input_layer/input_layer_1/pid_embedding/embedding_weights/part_0]input_layer/input_layer_1/pid_embedding/embedding_weights/part_0/Initializer/truncated_normal*S
_classI
GEloc:@input_layer/input_layer_1/pid_embedding/embedding_weights/part_0*
T0*
_output_shapes

:


Einput_layer/input_layer_1/pid_embedding/embedding_weights/part_0/readIdentity@input_layer/input_layer_1/pid_embedding/embedding_weights/part_0*
T0*
_output_shapes

:
*S
_classI
GEloc:@input_layer/input_layer_1/pid_embedding/embedding_weights/part_0

Iinput_layer/input_layer_1/pid_embedding/pid_embedding_weights/Slice/beginConst*
valueB: *
_output_shapes
:*
dtype0

Hinput_layer/input_layer_1/pid_embedding/pid_embedding_weights/Slice/sizeConst*
_output_shapes
:*
dtype0*
valueB:
Ř
Cinput_layer/input_layer_1/pid_embedding/pid_embedding_weights/SliceSliceCinput_layer/input_layer_1/pid_embedding/to_sparse_input/dense_shapeIinput_layer/input_layer_1/pid_embedding/pid_embedding_weights/Slice/beginHinput_layer/input_layer_1/pid_embedding/pid_embedding_weights/Slice/size*
_output_shapes
:*
Index0*
T0	

Cinput_layer/input_layer_1/pid_embedding/pid_embedding_weights/ConstConst*
_output_shapes
:*
valueB: *
dtype0
ő
Binput_layer/input_layer_1/pid_embedding/pid_embedding_weights/ProdProdCinput_layer/input_layer_1/pid_embedding/pid_embedding_weights/SliceCinput_layer/input_layer_1/pid_embedding/pid_embedding_weights/Const*
_output_shapes
: *
T0	

Ninput_layer/input_layer_1/pid_embedding/pid_embedding_weights/GatherV2/indicesConst*
value	B :*
dtype0*
_output_shapes
: 

Kinput_layer/input_layer_1/pid_embedding/pid_embedding_weights/GatherV2/axisConst*
value	B : *
_output_shapes
: *
dtype0
ř
Finput_layer/input_layer_1/pid_embedding/pid_embedding_weights/GatherV2GatherV2Cinput_layer/input_layer_1/pid_embedding/to_sparse_input/dense_shapeNinput_layer/input_layer_1/pid_embedding/pid_embedding_weights/GatherV2/indicesKinput_layer/input_layer_1/pid_embedding/pid_embedding_weights/GatherV2/axis*
Tindices0*
Taxis0*
Tparams0	*
_output_shapes
: 

Dinput_layer/input_layer_1/pid_embedding/pid_embedding_weights/Cast/xPackBinput_layer/input_layer_1/pid_embedding/pid_embedding_weights/ProdFinput_layer/input_layer_1/pid_embedding/pid_embedding_weights/GatherV2*
_output_shapes
:*
N*
T0	
×
Kinput_layer/input_layer_1/pid_embedding/pid_embedding_weights/SparseReshapeSparseReshape?input_layer/input_layer_1/pid_embedding/to_sparse_input/indicesCinput_layer/input_layer_1/pid_embedding/to_sparse_input/dense_shapeDinput_layer/input_layer_1/pid_embedding/pid_embedding_weights/Cast/x*-
_output_shapes
:˙˙˙˙˙˙˙˙˙:
ž
Tinput_layer/input_layer_1/pid_embedding/pid_embedding_weights/SparseReshape/IdentityIdentity.input_layer/input_layer_1/pid_embedding/lookup*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0	

Linput_layer/input_layer_1/pid_embedding/pid_embedding_weights/GreaterEqual/yConst*
_output_shapes
: *
dtype0	*
value	B	 R 
Ź
Jinput_layer/input_layer_1/pid_embedding/pid_embedding_weights/GreaterEqualGreaterEqualTinput_layer/input_layer_1/pid_embedding/pid_embedding_weights/SparseReshape/IdentityLinput_layer/input_layer_1/pid_embedding/pid_embedding_weights/GreaterEqual/y*
T0	*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
Á
Cinput_layer/input_layer_1/pid_embedding/pid_embedding_weights/WhereWhereJinput_layer/input_layer_1/pid_embedding/pid_embedding_weights/GreaterEqual*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

Kinput_layer/input_layer_1/pid_embedding/pid_embedding_weights/Reshape/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:

Einput_layer/input_layer_1/pid_embedding/pid_embedding_weights/ReshapeReshapeCinput_layer/input_layer_1/pid_embedding/pid_embedding_weights/WhereKinput_layer/input_layer_1/pid_embedding/pid_embedding_weights/Reshape/shape*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0	

Minput_layer/input_layer_1/pid_embedding/pid_embedding_weights/GatherV2_1/axisConst*
value	B : *
_output_shapes
: *
dtype0

Hinput_layer/input_layer_1/pid_embedding/pid_embedding_weights/GatherV2_1GatherV2Kinput_layer/input_layer_1/pid_embedding/pid_embedding_weights/SparseReshapeEinput_layer/input_layer_1/pid_embedding/pid_embedding_weights/ReshapeMinput_layer/input_layer_1/pid_embedding/pid_embedding_weights/GatherV2_1/axis*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
Tparams0	*
Taxis0*
Tindices0	

Minput_layer/input_layer_1/pid_embedding/pid_embedding_weights/GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
value	B : 

Hinput_layer/input_layer_1/pid_embedding/pid_embedding_weights/GatherV2_2GatherV2Tinput_layer/input_layer_1/pid_embedding/pid_embedding_weights/SparseReshape/IdentityEinput_layer/input_layer_1/pid_embedding/pid_embedding_weights/ReshapeMinput_layer/input_layer_1/pid_embedding/pid_embedding_weights/GatherV2_2/axis*
Tindices0	*
Tparams0	*
Taxis0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ć
Finput_layer/input_layer_1/pid_embedding/pid_embedding_weights/IdentityIdentityMinput_layer/input_layer_1/pid_embedding/pid_embedding_weights/SparseReshape:1*
T0	*
_output_shapes
:

Winput_layer/input_layer_1/pid_embedding/pid_embedding_weights/SparseFillEmptyRows/ConstConst*
value	B	 R *
dtype0	*
_output_shapes
: 

einput_layer/input_layer_1/pid_embedding/pid_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowsSparseFillEmptyRowsHinput_layer/input_layer_1/pid_embedding/pid_embedding_weights/GatherV2_1Hinput_layer/input_layer_1/pid_embedding/pid_embedding_weights/GatherV2_2Finput_layer/input_layer_1/pid_embedding/pid_embedding_weights/IdentityWinput_layer/input_layer_1/pid_embedding/pid_embedding_weights/SparseFillEmptyRows/Const*T
_output_shapesB
@:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0	
ş
iinput_layer/input_layer_1/pid_embedding/pid_embedding_weights/embedding_lookup_sparse/strided_slice/stackConst*
valueB"        *
dtype0*
_output_shapes
:
ź
kinput_layer/input_layer_1/pid_embedding/pid_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1Const*
valueB"       *
_output_shapes
:*
dtype0
ź
kinput_layer/input_layer_1/pid_embedding/pid_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2Const*
valueB"      *
_output_shapes
:*
dtype0

cinput_layer/input_layer_1/pid_embedding/pid_embedding_weights/embedding_lookup_sparse/strided_sliceStridedSliceeinput_layer/input_layer_1/pid_embedding/pid_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowsiinput_layer/input_layer_1/pid_embedding/pid_embedding_weights/embedding_lookup_sparse/strided_slice/stackkinput_layer/input_layer_1/pid_embedding/pid_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1kinput_layer/input_layer_1/pid_embedding/pid_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2*
T0	*

begin_mask*
Index0*
shrink_axis_mask*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
end_mask

Zinput_layer/input_layer_1/pid_embedding/pid_embedding_weights/embedding_lookup_sparse/CastCastcinput_layer/input_layer_1/pid_embedding/pid_embedding_weights/embedding_lookup_sparse/strided_slice*

SrcT0	*

DstT0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙

\input_layer/input_layer_1/pid_embedding/pid_embedding_weights/embedding_lookup_sparse/UniqueUniqueginput_layer/input_layer_1/pid_embedding/pid_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0	

kinput_layer/input_layer_1/pid_embedding/pid_embedding_weights/embedding_lookup_sparse/embedding_lookup/axisConst*
_output_shapes
: *S
_classI
GEloc:@input_layer/input_layer_1/pid_embedding/embedding_weights/part_0*
value	B : *
dtype0
Ž
finput_layer/input_layer_1/pid_embedding/pid_embedding_weights/embedding_lookup_sparse/embedding_lookupGatherV2Einput_layer/input_layer_1/pid_embedding/embedding_weights/part_0/read\input_layer/input_layer_1/pid_embedding/pid_embedding_weights/embedding_lookup_sparse/Uniquekinput_layer/input_layer_1/pid_embedding/pid_embedding_weights/embedding_lookup_sparse/embedding_lookup/axis*S
_classI
GEloc:@input_layer/input_layer_1/pid_embedding/embedding_weights/part_0*
Tindices0	*
Taxis0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
Tparams0

oinput_layer/input_layer_1/pid_embedding/pid_embedding_weights/embedding_lookup_sparse/embedding_lookup/IdentityIdentityfinput_layer/input_layer_1/pid_embedding/pid_embedding_weights/embedding_lookup_sparse/embedding_lookup*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
É
Uinput_layer/input_layer_1/pid_embedding/pid_embedding_weights/embedding_lookup_sparseSparseSegmentMeanoinput_layer/input_layer_1/pid_embedding/pid_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identity^input_layer/input_layer_1/pid_embedding/pid_embedding_weights/embedding_lookup_sparse/Unique:1Zinput_layer/input_layer_1/pid_embedding/pid_embedding_weights/embedding_lookup_sparse/Cast*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

Minput_layer/input_layer_1/pid_embedding/pid_embedding_weights/Reshape_1/shapeConst*
dtype0*
_output_shapes
:*
valueB"˙˙˙˙   
ź
Ginput_layer/input_layer_1/pid_embedding/pid_embedding_weights/Reshape_1Reshapeginput_layer/input_layer_1/pid_embedding/pid_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:2Minput_layer/input_layer_1/pid_embedding/pid_embedding_weights/Reshape_1/shape*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

Č
Cinput_layer/input_layer_1/pid_embedding/pid_embedding_weights/ShapeShapeUinput_layer/input_layer_1/pid_embedding/pid_embedding_weights/embedding_lookup_sparse*
_output_shapes
:*
T0

Qinput_layer/input_layer_1/pid_embedding/pid_embedding_weights/strided_slice/stackConst*
valueB:*
_output_shapes
:*
dtype0

Sinput_layer/input_layer_1/pid_embedding/pid_embedding_weights/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:

Sinput_layer/input_layer_1/pid_embedding/pid_embedding_weights/strided_slice/stack_2Const*
_output_shapes
:*
valueB:*
dtype0
ă
Kinput_layer/input_layer_1/pid_embedding/pid_embedding_weights/strided_sliceStridedSliceCinput_layer/input_layer_1/pid_embedding/pid_embedding_weights/ShapeQinput_layer/input_layer_1/pid_embedding/pid_embedding_weights/strided_slice/stackSinput_layer/input_layer_1/pid_embedding/pid_embedding_weights/strided_slice/stack_1Sinput_layer/input_layer_1/pid_embedding/pid_embedding_weights/strided_slice/stack_2*
T0*
Index0*
_output_shapes
: *
shrink_axis_mask

Einput_layer/input_layer_1/pid_embedding/pid_embedding_weights/stack/0Const*
_output_shapes
: *
dtype0*
value	B :

Cinput_layer/input_layer_1/pid_embedding/pid_embedding_weights/stackPackEinput_layer/input_layer_1/pid_embedding/pid_embedding_weights/stack/0Kinput_layer/input_layer_1/pid_embedding/pid_embedding_weights/strided_slice*
_output_shapes
:*
N*
T0

Binput_layer/input_layer_1/pid_embedding/pid_embedding_weights/TileTileGinput_layer/input_layer_1/pid_embedding/pid_embedding_weights/Reshape_1Cinput_layer/input_layer_1/pid_embedding/pid_embedding_weights/stack*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
T0

Ţ
Hinput_layer/input_layer_1/pid_embedding/pid_embedding_weights/zeros_like	ZerosLikeUinput_layer/input_layer_1/pid_embedding/pid_embedding_weights/embedding_lookup_sparse*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ţ
=input_layer/input_layer_1/pid_embedding/pid_embedding_weightsSelectBinput_layer/input_layer_1/pid_embedding/pid_embedding_weights/TileHinput_layer/input_layer_1/pid_embedding/pid_embedding_weights/zeros_likeUinput_layer/input_layer_1/pid_embedding/pid_embedding_weights/embedding_lookup_sparse*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ĺ
Dinput_layer/input_layer_1/pid_embedding/pid_embedding_weights/Cast_1CastCinput_layer/input_layer_1/pid_embedding/to_sparse_input/dense_shape*
_output_shapes
:*

SrcT0	*

DstT0

Kinput_layer/input_layer_1/pid_embedding/pid_embedding_weights/Slice_1/beginConst*
valueB: *
_output_shapes
:*
dtype0

Jinput_layer/input_layer_1/pid_embedding/pid_embedding_weights/Slice_1/sizeConst*
valueB:*
_output_shapes
:*
dtype0
ß
Einput_layer/input_layer_1/pid_embedding/pid_embedding_weights/Slice_1SliceDinput_layer/input_layer_1/pid_embedding/pid_embedding_weights/Cast_1Kinput_layer/input_layer_1/pid_embedding/pid_embedding_weights/Slice_1/beginJinput_layer/input_layer_1/pid_embedding/pid_embedding_weights/Slice_1/size*
Index0*
_output_shapes
:*
T0
˛
Einput_layer/input_layer_1/pid_embedding/pid_embedding_weights/Shape_1Shape=input_layer/input_layer_1/pid_embedding/pid_embedding_weights*
T0*
_output_shapes
:

Kinput_layer/input_layer_1/pid_embedding/pid_embedding_weights/Slice_2/beginConst*
valueB:*
dtype0*
_output_shapes
:

Jinput_layer/input_layer_1/pid_embedding/pid_embedding_weights/Slice_2/sizeConst*
dtype0*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙
ŕ
Einput_layer/input_layer_1/pid_embedding/pid_embedding_weights/Slice_2SliceEinput_layer/input_layer_1/pid_embedding/pid_embedding_weights/Shape_1Kinput_layer/input_layer_1/pid_embedding/pid_embedding_weights/Slice_2/beginJinput_layer/input_layer_1/pid_embedding/pid_embedding_weights/Slice_2/size*
Index0*
T0*
_output_shapes
:

Iinput_layer/input_layer_1/pid_embedding/pid_embedding_weights/concat/axisConst*
dtype0*
value	B : *
_output_shapes
: 
×
Dinput_layer/input_layer_1/pid_embedding/pid_embedding_weights/concatConcatV2Einput_layer/input_layer_1/pid_embedding/pid_embedding_weights/Slice_1Einput_layer/input_layer_1/pid_embedding/pid_embedding_weights/Slice_2Iinput_layer/input_layer_1/pid_embedding/pid_embedding_weights/concat/axis*
N*
_output_shapes
:*
T0

Ginput_layer/input_layer_1/pid_embedding/pid_embedding_weights/Reshape_2Reshape=input_layer/input_layer_1/pid_embedding/pid_embedding_weightsDinput_layer/input_layer_1/pid_embedding/pid_embedding_weights/concat*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
¤
-input_layer/input_layer_1/pid_embedding/ShapeShapeGinput_layer/input_layer_1/pid_embedding/pid_embedding_weights/Reshape_2*
_output_shapes
:*
T0

;input_layer/input_layer_1/pid_embedding/strided_slice/stackConst*
_output_shapes
:*
valueB: *
dtype0

=input_layer/input_layer_1/pid_embedding/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:

=input_layer/input_layer_1/pid_embedding/strided_slice/stack_2Const*
valueB:*
_output_shapes
:*
dtype0
ő
5input_layer/input_layer_1/pid_embedding/strided_sliceStridedSlice-input_layer/input_layer_1/pid_embedding/Shape;input_layer/input_layer_1/pid_embedding/strided_slice/stack=input_layer/input_layer_1/pid_embedding/strided_slice/stack_1=input_layer/input_layer_1/pid_embedding/strided_slice/stack_2*
shrink_axis_mask*
_output_shapes
: *
Index0*
T0
y
7input_layer/input_layer_1/pid_embedding/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
Ű
5input_layer/input_layer_1/pid_embedding/Reshape/shapePack5input_layer/input_layer_1/pid_embedding/strided_slice7input_layer/input_layer_1/pid_embedding/Reshape/shape/1*
T0*
_output_shapes
:*
N
ě
/input_layer/input_layer_1/pid_embedding/ReshapeReshapeGinput_layer/input_layer_1/pid_embedding/pid_embedding_weights/Reshape_25input_layer/input_layer_1/pid_embedding/Reshape/shape*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

8input_layer/input_layer_1/price_embedding/ExpandDims/dimConst*
valueB :
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
: 
ž
4input_layer/input_layer_1/price_embedding/ExpandDims
ExpandDimsPlaceholder_178input_layer/input_layer_1/price_embedding/ExpandDims/dim*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

Hinput_layer/input_layer_1/price_embedding/to_sparse_input/ignore_value/xConst*
dtype0*
_output_shapes
: *
valueB B 

Binput_layer/input_layer_1/price_embedding/to_sparse_input/NotEqualNotEqual4input_layer/input_layer_1/price_embedding/ExpandDimsHinput_layer/input_layer_1/price_embedding/to_sparse_input/ignore_value/x*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
ˇ
Ainput_layer/input_layer_1/price_embedding/to_sparse_input/indicesWhereBinput_layer/input_layer_1/price_embedding/to_sparse_input/NotEqual*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

@input_layer/input_layer_1/price_embedding/to_sparse_input/valuesGatherNd4input_layer/input_layer_1/price_embedding/ExpandDimsAinput_layer/input_layer_1/price_embedding/to_sparse_input/indices*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
Tindices0	*
Tparams0
š
Einput_layer/input_layer_1/price_embedding/to_sparse_input/dense_shapeShape4input_layer/input_layer_1/price_embedding/ExpandDims*
_output_shapes
:*
T0*
out_type0	
Ä
0input_layer/input_layer_1/price_embedding/lookupStringToHashBucketFast@input_layer/input_layer_1/price_embedding/to_sparse_input/values*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
num_buckets2

einput_layer/input_layer_1/price_embedding/embedding_weights/part_0/Initializer/truncated_normal/shapeConst*
_output_shapes
:*
valueB"2      *
dtype0*U
_classK
IGloc:@input_layer/input_layer_1/price_embedding/embedding_weights/part_0

dinput_layer/input_layer_1/price_embedding/embedding_weights/part_0/Initializer/truncated_normal/meanConst*U
_classK
IGloc:@input_layer/input_layer_1/price_embedding/embedding_weights/part_0*
valueB
 *    *
_output_shapes
: *
dtype0

finput_layer/input_layer_1/price_embedding/embedding_weights/part_0/Initializer/truncated_normal/stddevConst*
dtype0*
valueB
 *  >*U
_classK
IGloc:@input_layer/input_layer_1/price_embedding/embedding_weights/part_0*
_output_shapes
: 
ö
oinput_layer/input_layer_1/price_embedding/embedding_weights/part_0/Initializer/truncated_normal/TruncatedNormalTruncatedNormaleinput_layer/input_layer_1/price_embedding/embedding_weights/part_0/Initializer/truncated_normal/shape*
_output_shapes

:2*U
_classK
IGloc:@input_layer/input_layer_1/price_embedding/embedding_weights/part_0*
T0*
dtype0
Ă
cinput_layer/input_layer_1/price_embedding/embedding_weights/part_0/Initializer/truncated_normal/mulMuloinput_layer/input_layer_1/price_embedding/embedding_weights/part_0/Initializer/truncated_normal/TruncatedNormalfinput_layer/input_layer_1/price_embedding/embedding_weights/part_0/Initializer/truncated_normal/stddev*
_output_shapes

:2*U
_classK
IGloc:@input_layer/input_layer_1/price_embedding/embedding_weights/part_0*
T0
ą
_input_layer/input_layer_1/price_embedding/embedding_weights/part_0/Initializer/truncated_normalAddcinput_layer/input_layer_1/price_embedding/embedding_weights/part_0/Initializer/truncated_normal/muldinput_layer/input_layer_1/price_embedding/embedding_weights/part_0/Initializer/truncated_normal/mean*
T0*
_output_shapes

:2*U
_classK
IGloc:@input_layer/input_layer_1/price_embedding/embedding_weights/part_0
é
Binput_layer/input_layer_1/price_embedding/embedding_weights/part_0
VariableV2*
shape
:2*U
_classK
IGloc:@input_layer/input_layer_1/price_embedding/embedding_weights/part_0*
_output_shapes

:2*
dtype0
ř
Iinput_layer/input_layer_1/price_embedding/embedding_weights/part_0/AssignAssignBinput_layer/input_layer_1/price_embedding/embedding_weights/part_0_input_layer/input_layer_1/price_embedding/embedding_weights/part_0/Initializer/truncated_normal*
T0*
_output_shapes

:2*U
_classK
IGloc:@input_layer/input_layer_1/price_embedding/embedding_weights/part_0

Ginput_layer/input_layer_1/price_embedding/embedding_weights/part_0/readIdentityBinput_layer/input_layer_1/price_embedding/embedding_weights/part_0*
T0*U
_classK
IGloc:@input_layer/input_layer_1/price_embedding/embedding_weights/part_0*
_output_shapes

:2

Minput_layer/input_layer_1/price_embedding/price_embedding_weights/Slice/beginConst*
_output_shapes
:*
dtype0*
valueB: 

Linput_layer/input_layer_1/price_embedding/price_embedding_weights/Slice/sizeConst*
valueB:*
dtype0*
_output_shapes
:
ć
Ginput_layer/input_layer_1/price_embedding/price_embedding_weights/SliceSliceEinput_layer/input_layer_1/price_embedding/to_sparse_input/dense_shapeMinput_layer/input_layer_1/price_embedding/price_embedding_weights/Slice/beginLinput_layer/input_layer_1/price_embedding/price_embedding_weights/Slice/size*
T0	*
_output_shapes
:*
Index0

Ginput_layer/input_layer_1/price_embedding/price_embedding_weights/ConstConst*
_output_shapes
:*
valueB: *
dtype0

Finput_layer/input_layer_1/price_embedding/price_embedding_weights/ProdProdGinput_layer/input_layer_1/price_embedding/price_embedding_weights/SliceGinput_layer/input_layer_1/price_embedding/price_embedding_weights/Const*
_output_shapes
: *
T0	

Rinput_layer/input_layer_1/price_embedding/price_embedding_weights/GatherV2/indicesConst*
dtype0*
value	B :*
_output_shapes
: 

Oinput_layer/input_layer_1/price_embedding/price_embedding_weights/GatherV2/axisConst*
value	B : *
_output_shapes
: *
dtype0

Jinput_layer/input_layer_1/price_embedding/price_embedding_weights/GatherV2GatherV2Einput_layer/input_layer_1/price_embedding/to_sparse_input/dense_shapeRinput_layer/input_layer_1/price_embedding/price_embedding_weights/GatherV2/indicesOinput_layer/input_layer_1/price_embedding/price_embedding_weights/GatherV2/axis*
Taxis0*
Tparams0	*
Tindices0*
_output_shapes
: 

Hinput_layer/input_layer_1/price_embedding/price_embedding_weights/Cast/xPackFinput_layer/input_layer_1/price_embedding/price_embedding_weights/ProdJinput_layer/input_layer_1/price_embedding/price_embedding_weights/GatherV2*
_output_shapes
:*
T0	*
N
ă
Oinput_layer/input_layer_1/price_embedding/price_embedding_weights/SparseReshapeSparseReshapeAinput_layer/input_layer_1/price_embedding/to_sparse_input/indicesEinput_layer/input_layer_1/price_embedding/to_sparse_input/dense_shapeHinput_layer/input_layer_1/price_embedding/price_embedding_weights/Cast/x*-
_output_shapes
:˙˙˙˙˙˙˙˙˙:
Ä
Xinput_layer/input_layer_1/price_embedding/price_embedding_weights/SparseReshape/IdentityIdentity0input_layer/input_layer_1/price_embedding/lookup*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0	

Pinput_layer/input_layer_1/price_embedding/price_embedding_weights/GreaterEqual/yConst*
dtype0	*
_output_shapes
: *
value	B	 R 
¸
Ninput_layer/input_layer_1/price_embedding/price_embedding_weights/GreaterEqualGreaterEqualXinput_layer/input_layer_1/price_embedding/price_embedding_weights/SparseReshape/IdentityPinput_layer/input_layer_1/price_embedding/price_embedding_weights/GreaterEqual/y*
T0	*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
É
Ginput_layer/input_layer_1/price_embedding/price_embedding_weights/WhereWhereNinput_layer/input_layer_1/price_embedding/price_embedding_weights/GreaterEqual*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
˘
Oinput_layer/input_layer_1/price_embedding/price_embedding_weights/Reshape/shapeConst*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙*
_output_shapes
:

Iinput_layer/input_layer_1/price_embedding/price_embedding_weights/ReshapeReshapeGinput_layer/input_layer_1/price_embedding/price_embedding_weights/WhereOinput_layer/input_layer_1/price_embedding/price_embedding_weights/Reshape/shape*
T0	*#
_output_shapes
:˙˙˙˙˙˙˙˙˙

Qinput_layer/input_layer_1/price_embedding/price_embedding_weights/GatherV2_1/axisConst*
value	B : *
_output_shapes
: *
dtype0

Linput_layer/input_layer_1/price_embedding/price_embedding_weights/GatherV2_1GatherV2Oinput_layer/input_layer_1/price_embedding/price_embedding_weights/SparseReshapeIinput_layer/input_layer_1/price_embedding/price_embedding_weights/ReshapeQinput_layer/input_layer_1/price_embedding/price_embedding_weights/GatherV2_1/axis*
Taxis0*
Tindices0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
Tparams0	

Qinput_layer/input_layer_1/price_embedding/price_embedding_weights/GatherV2_2/axisConst*
dtype0*
value	B : *
_output_shapes
: 
Ą
Linput_layer/input_layer_1/price_embedding/price_embedding_weights/GatherV2_2GatherV2Xinput_layer/input_layer_1/price_embedding/price_embedding_weights/SparseReshape/IdentityIinput_layer/input_layer_1/price_embedding/price_embedding_weights/ReshapeQinput_layer/input_layer_1/price_embedding/price_embedding_weights/GatherV2_2/axis*
Tindices0	*
Tparams0	*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
Taxis0
Î
Jinput_layer/input_layer_1/price_embedding/price_embedding_weights/IdentityIdentityQinput_layer/input_layer_1/price_embedding/price_embedding_weights/SparseReshape:1*
T0	*
_output_shapes
:

[input_layer/input_layer_1/price_embedding/price_embedding_weights/SparseFillEmptyRows/ConstConst*
_output_shapes
: *
value	B	 R *
dtype0	
¤
iinput_layer/input_layer_1/price_embedding/price_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowsSparseFillEmptyRowsLinput_layer/input_layer_1/price_embedding/price_embedding_weights/GatherV2_1Linput_layer/input_layer_1/price_embedding/price_embedding_weights/GatherV2_2Jinput_layer/input_layer_1/price_embedding/price_embedding_weights/Identity[input_layer/input_layer_1/price_embedding/price_embedding_weights/SparseFillEmptyRows/Const*
T0	*T
_output_shapesB
@:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
ž
minput_layer/input_layer_1/price_embedding/price_embedding_weights/embedding_lookup_sparse/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB"        
Ŕ
oinput_layer/input_layer_1/price_embedding/price_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1Const*
_output_shapes
:*
valueB"       *
dtype0
Ŕ
oinput_layer/input_layer_1/price_embedding/price_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB"      
¨
ginput_layer/input_layer_1/price_embedding/price_embedding_weights/embedding_lookup_sparse/strided_sliceStridedSliceiinput_layer/input_layer_1/price_embedding/price_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowsminput_layer/input_layer_1/price_embedding/price_embedding_weights/embedding_lookup_sparse/strided_slice/stackoinput_layer/input_layer_1/price_embedding/price_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1oinput_layer/input_layer_1/price_embedding/price_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2*
shrink_axis_mask*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
Index0*
T0	*

begin_mask*
end_mask

^input_layer/input_layer_1/price_embedding/price_embedding_weights/embedding_lookup_sparse/CastCastginput_layer/input_layer_1/price_embedding/price_embedding_weights/embedding_lookup_sparse/strided_slice*

SrcT0	*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*

DstT0

`input_layer/input_layer_1/price_embedding/price_embedding_weights/embedding_lookup_sparse/UniqueUniquekinput_layer/input_layer_1/price_embedding/price_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0	

oinput_layer/input_layer_1/price_embedding/price_embedding_weights/embedding_lookup_sparse/embedding_lookup/axisConst*U
_classK
IGloc:@input_layer/input_layer_1/price_embedding/embedding_weights/part_0*
value	B : *
_output_shapes
: *
dtype0
ž
jinput_layer/input_layer_1/price_embedding/price_embedding_weights/embedding_lookup_sparse/embedding_lookupGatherV2Ginput_layer/input_layer_1/price_embedding/embedding_weights/part_0/read`input_layer/input_layer_1/price_embedding/price_embedding_weights/embedding_lookup_sparse/Uniqueoinput_layer/input_layer_1/price_embedding/price_embedding_weights/embedding_lookup_sparse/embedding_lookup/axis*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
Tparams0*
Tindices0	*U
_classK
IGloc:@input_layer/input_layer_1/price_embedding/embedding_weights/part_0*
Taxis0

sinput_layer/input_layer_1/price_embedding/price_embedding_weights/embedding_lookup_sparse/embedding_lookup/IdentityIdentityjinput_layer/input_layer_1/price_embedding/price_embedding_weights/embedding_lookup_sparse/embedding_lookup*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ů
Yinput_layer/input_layer_1/price_embedding/price_embedding_weights/embedding_lookup_sparseSparseSegmentMeansinput_layer/input_layer_1/price_embedding/price_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identitybinput_layer/input_layer_1/price_embedding/price_embedding_weights/embedding_lookup_sparse/Unique:1^input_layer/input_layer_1/price_embedding/price_embedding_weights/embedding_lookup_sparse/Cast*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
˘
Qinput_layer/input_layer_1/price_embedding/price_embedding_weights/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   
Č
Kinput_layer/input_layer_1/price_embedding/price_embedding_weights/Reshape_1Reshapekinput_layer/input_layer_1/price_embedding/price_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:2Qinput_layer/input_layer_1/price_embedding/price_embedding_weights/Reshape_1/shape*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

Đ
Ginput_layer/input_layer_1/price_embedding/price_embedding_weights/ShapeShapeYinput_layer/input_layer_1/price_embedding/price_embedding_weights/embedding_lookup_sparse*
T0*
_output_shapes
:

Uinput_layer/input_layer_1/price_embedding/price_embedding_weights/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
Ą
Winput_layer/input_layer_1/price_embedding/price_embedding_weights/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
Ą
Winput_layer/input_layer_1/price_embedding/price_embedding_weights/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
÷
Oinput_layer/input_layer_1/price_embedding/price_embedding_weights/strided_sliceStridedSliceGinput_layer/input_layer_1/price_embedding/price_embedding_weights/ShapeUinput_layer/input_layer_1/price_embedding/price_embedding_weights/strided_slice/stackWinput_layer/input_layer_1/price_embedding/price_embedding_weights/strided_slice/stack_1Winput_layer/input_layer_1/price_embedding/price_embedding_weights/strided_slice/stack_2*
_output_shapes
: *
T0*
shrink_axis_mask*
Index0

Iinput_layer/input_layer_1/price_embedding/price_embedding_weights/stack/0Const*
dtype0*
_output_shapes
: *
value	B :

Ginput_layer/input_layer_1/price_embedding/price_embedding_weights/stackPackIinput_layer/input_layer_1/price_embedding/price_embedding_weights/stack/0Oinput_layer/input_layer_1/price_embedding/price_embedding_weights/strided_slice*
_output_shapes
:*
T0*
N

Finput_layer/input_layer_1/price_embedding/price_embedding_weights/TileTileKinput_layer/input_layer_1/price_embedding/price_embedding_weights/Reshape_1Ginput_layer/input_layer_1/price_embedding/price_embedding_weights/stack*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
T0

ć
Linput_layer/input_layer_1/price_embedding/price_embedding_weights/zeros_like	ZerosLikeYinput_layer/input_layer_1/price_embedding/price_embedding_weights/embedding_lookup_sparse*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
î
Ainput_layer/input_layer_1/price_embedding/price_embedding_weightsSelectFinput_layer/input_layer_1/price_embedding/price_embedding_weights/TileLinput_layer/input_layer_1/price_embedding/price_embedding_weights/zeros_likeYinput_layer/input_layer_1/price_embedding/price_embedding_weights/embedding_lookup_sparse*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ë
Hinput_layer/input_layer_1/price_embedding/price_embedding_weights/Cast_1CastEinput_layer/input_layer_1/price_embedding/to_sparse_input/dense_shape*

SrcT0	*
_output_shapes
:*

DstT0

Oinput_layer/input_layer_1/price_embedding/price_embedding_weights/Slice_1/beginConst*
dtype0*
_output_shapes
:*
valueB: 

Ninput_layer/input_layer_1/price_embedding/price_embedding_weights/Slice_1/sizeConst*
valueB:*
dtype0*
_output_shapes
:
ď
Iinput_layer/input_layer_1/price_embedding/price_embedding_weights/Slice_1SliceHinput_layer/input_layer_1/price_embedding/price_embedding_weights/Cast_1Oinput_layer/input_layer_1/price_embedding/price_embedding_weights/Slice_1/beginNinput_layer/input_layer_1/price_embedding/price_embedding_weights/Slice_1/size*
T0*
Index0*
_output_shapes
:
ş
Iinput_layer/input_layer_1/price_embedding/price_embedding_weights/Shape_1ShapeAinput_layer/input_layer_1/price_embedding/price_embedding_weights*
T0*
_output_shapes
:

Oinput_layer/input_layer_1/price_embedding/price_embedding_weights/Slice_2/beginConst*
dtype0*
_output_shapes
:*
valueB:
Ą
Ninput_layer/input_layer_1/price_embedding/price_embedding_weights/Slice_2/sizeConst*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙*
_output_shapes
:
đ
Iinput_layer/input_layer_1/price_embedding/price_embedding_weights/Slice_2SliceIinput_layer/input_layer_1/price_embedding/price_embedding_weights/Shape_1Oinput_layer/input_layer_1/price_embedding/price_embedding_weights/Slice_2/beginNinput_layer/input_layer_1/price_embedding/price_embedding_weights/Slice_2/size*
T0*
Index0*
_output_shapes
:

Minput_layer/input_layer_1/price_embedding/price_embedding_weights/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
ç
Hinput_layer/input_layer_1/price_embedding/price_embedding_weights/concatConcatV2Iinput_layer/input_layer_1/price_embedding/price_embedding_weights/Slice_1Iinput_layer/input_layer_1/price_embedding/price_embedding_weights/Slice_2Minput_layer/input_layer_1/price_embedding/price_embedding_weights/concat/axis*
_output_shapes
:*
N*
T0

Kinput_layer/input_layer_1/price_embedding/price_embedding_weights/Reshape_2ReshapeAinput_layer/input_layer_1/price_embedding/price_embedding_weightsHinput_layer/input_layer_1/price_embedding/price_embedding_weights/concat*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ş
/input_layer/input_layer_1/price_embedding/ShapeShapeKinput_layer/input_layer_1/price_embedding/price_embedding_weights/Reshape_2*
T0*
_output_shapes
:

=input_layer/input_layer_1/price_embedding/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 

?input_layer/input_layer_1/price_embedding/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:

?input_layer/input_layer_1/price_embedding/strided_slice/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
˙
7input_layer/input_layer_1/price_embedding/strided_sliceStridedSlice/input_layer/input_layer_1/price_embedding/Shape=input_layer/input_layer_1/price_embedding/strided_slice/stack?input_layer/input_layer_1/price_embedding/strided_slice/stack_1?input_layer/input_layer_1/price_embedding/strided_slice/stack_2*
T0*
_output_shapes
: *
shrink_axis_mask*
Index0
{
9input_layer/input_layer_1/price_embedding/Reshape/shape/1Const*
value	B :*
_output_shapes
: *
dtype0
á
7input_layer/input_layer_1/price_embedding/Reshape/shapePack7input_layer/input_layer_1/price_embedding/strided_slice9input_layer/input_layer_1/price_embedding/Reshape/shape/1*
N*
T0*
_output_shapes
:
ô
1input_layer/input_layer_1/price_embedding/ReshapeReshapeKinput_layer/input_layer_1/price_embedding/price_embedding_weights/Reshape_27input_layer/input_layer_1/price_embedding/Reshape/shape*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
g
%input_layer/input_layer_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :
Á
 input_layer/input_layer_1/concatConcatV26input_layer/input_layer_1/adgroup_id_embedding/Reshape1input_layer/input_layer_1/brand_embedding/Reshape7input_layer/input_layer_1/campaign_id_embedding/Reshape3input_layer/input_layer_1/cate_id_embedding/Reshape4input_layer/input_layer_1/customer_embedding/Reshape=input_layer/input_layer_1/final_gender_code_embedding/Reshape/input_layer/input_layer_1/pid_embedding/Reshape1input_layer/input_layer_1/price_embedding/Reshape%input_layer/input_layer_1/concat/axis*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
N
˙
]user_dnn_layer/user_dnn_layer/user_dnn_0/dense/kernel/part_0/Initializer/random_uniform/shapeConst*
dtype0*
valueB"      *
_output_shapes
:*O
_classE
CAloc:@user_dnn_layer/user_dnn_layer/user_dnn_0/dense/kernel/part_0
ń
[user_dnn_layer/user_dnn_layer/user_dnn_0/dense/kernel/part_0/Initializer/random_uniform/minConst*
valueB
 *   ž*
_output_shapes
: *
dtype0*O
_classE
CAloc:@user_dnn_layer/user_dnn_layer/user_dnn_0/dense/kernel/part_0
ń
[user_dnn_layer/user_dnn_layer/user_dnn_0/dense/kernel/part_0/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *O
_classE
CAloc:@user_dnn_layer/user_dnn_layer/user_dnn_0/dense/kernel/part_0*
valueB
 *   >
Ţ
euser_dnn_layer/user_dnn_layer/user_dnn_0/dense/kernel/part_0/Initializer/random_uniform/RandomUniformRandomUniform]user_dnn_layer/user_dnn_layer/user_dnn_0/dense/kernel/part_0/Initializer/random_uniform/shape*
T0*O
_classE
CAloc:@user_dnn_layer/user_dnn_layer/user_dnn_0/dense/kernel/part_0*
dtype0* 
_output_shapes
:


[user_dnn_layer/user_dnn_layer/user_dnn_0/dense/kernel/part_0/Initializer/random_uniform/subSub[user_dnn_layer/user_dnn_layer/user_dnn_0/dense/kernel/part_0/Initializer/random_uniform/max[user_dnn_layer/user_dnn_layer/user_dnn_0/dense/kernel/part_0/Initializer/random_uniform/min*
_output_shapes
: *
T0*O
_classE
CAloc:@user_dnn_layer/user_dnn_layer/user_dnn_0/dense/kernel/part_0
˘
[user_dnn_layer/user_dnn_layer/user_dnn_0/dense/kernel/part_0/Initializer/random_uniform/mulMuleuser_dnn_layer/user_dnn_layer/user_dnn_0/dense/kernel/part_0/Initializer/random_uniform/RandomUniform[user_dnn_layer/user_dnn_layer/user_dnn_0/dense/kernel/part_0/Initializer/random_uniform/sub*O
_classE
CAloc:@user_dnn_layer/user_dnn_layer/user_dnn_0/dense/kernel/part_0*
T0* 
_output_shapes
:


Wuser_dnn_layer/user_dnn_layer/user_dnn_0/dense/kernel/part_0/Initializer/random_uniformAdd[user_dnn_layer/user_dnn_layer/user_dnn_0/dense/kernel/part_0/Initializer/random_uniform/mul[user_dnn_layer/user_dnn_layer/user_dnn_0/dense/kernel/part_0/Initializer/random_uniform/min*O
_classE
CAloc:@user_dnn_layer/user_dnn_layer/user_dnn_0/dense/kernel/part_0*
T0* 
_output_shapes
:

á
<user_dnn_layer/user_dnn_layer/user_dnn_0/dense/kernel/part_0
VariableV2*
dtype0* 
_output_shapes
:
*O
_classE
CAloc:@user_dnn_layer/user_dnn_layer/user_dnn_0/dense/kernel/part_0*
shape:

ŕ
Cuser_dnn_layer/user_dnn_layer/user_dnn_0/dense/kernel/part_0/AssignAssign<user_dnn_layer/user_dnn_layer/user_dnn_0/dense/kernel/part_0Wuser_dnn_layer/user_dnn_layer/user_dnn_0/dense/kernel/part_0/Initializer/random_uniform* 
_output_shapes
:
*O
_classE
CAloc:@user_dnn_layer/user_dnn_layer/user_dnn_0/dense/kernel/part_0*
T0

Auser_dnn_layer/user_dnn_layer/user_dnn_0/dense/kernel/part_0/readIdentity<user_dnn_layer/user_dnn_layer/user_dnn_0/dense/kernel/part_0* 
_output_shapes
:
*O
_classE
CAloc:@user_dnn_layer/user_dnn_layer/user_dnn_0/dense/kernel/part_0*
T0
ę
Luser_dnn_layer/user_dnn_layer/user_dnn_0/dense/bias/part_0/Initializer/zerosConst*
dtype0*
_output_shapes	
:*
valueB*    *M
_classC
A?loc:@user_dnn_layer/user_dnn_layer/user_dnn_0/dense/bias/part_0
Ó
:user_dnn_layer/user_dnn_layer/user_dnn_0/dense/bias/part_0
VariableV2*M
_classC
A?loc:@user_dnn_layer/user_dnn_layer/user_dnn_0/dense/bias/part_0*
shape:*
dtype0*
_output_shapes	
:
Ę
Auser_dnn_layer/user_dnn_layer/user_dnn_0/dense/bias/part_0/AssignAssign:user_dnn_layer/user_dnn_layer/user_dnn_0/dense/bias/part_0Luser_dnn_layer/user_dnn_layer/user_dnn_0/dense/bias/part_0/Initializer/zeros*
_output_shapes	
:*
T0*M
_classC
A?loc:@user_dnn_layer/user_dnn_layer/user_dnn_0/dense/bias/part_0
ü
?user_dnn_layer/user_dnn_layer/user_dnn_0/dense/bias/part_0/readIdentity:user_dnn_layer/user_dnn_layer/user_dnn_0/dense/bias/part_0*
_output_shapes	
:*
T0*M
_classC
A?loc:@user_dnn_layer/user_dnn_layer/user_dnn_0/dense/bias/part_0
Ż
5user_dnn_layer/user_dnn_layer/user_dnn_0/dense/kernelIdentityAuser_dnn_layer/user_dnn_layer/user_dnn_0/dense/kernel/part_0/read* 
_output_shapes
:
*
T0
É
5user_dnn_layer/user_dnn_layer/user_dnn_0/dense/MatMulMatMulinput_layer/input_layer/concat5user_dnn_layer/user_dnn_layer/user_dnn_0/dense/kernel*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ś
3user_dnn_layer/user_dnn_layer/user_dnn_0/dense/biasIdentity?user_dnn_layer/user_dnn_layer/user_dnn_0/dense/bias/part_0/read*
_output_shapes	
:*
T0
ŕ
6user_dnn_layer/user_dnn_layer/user_dnn_0/dense/BiasAddBiasAdd5user_dnn_layer/user_dnn_layer/user_dnn_0/dense/MatMul3user_dnn_layer/user_dnn_layer/user_dnn_0/dense/bias*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

Zuser_dnn_layer/user_dnn_layer/user_dnn_0/batch_normalization/gamma/part_0/Initializer/onesConst*\
_classR
PNloc:@user_dnn_layer/user_dnn_layer/user_dnn_0/batch_normalization/gamma/part_0*
_output_shapes	
:*
valueB*  ?*
dtype0
ń
Iuser_dnn_layer/user_dnn_layer/user_dnn_0/batch_normalization/gamma/part_0
VariableV2*\
_classR
PNloc:@user_dnn_layer/user_dnn_layer/user_dnn_0/batch_normalization/gamma/part_0*
dtype0*
shape:*
_output_shapes	
:

Puser_dnn_layer/user_dnn_layer/user_dnn_0/batch_normalization/gamma/part_0/AssignAssignIuser_dnn_layer/user_dnn_layer/user_dnn_0/batch_normalization/gamma/part_0Zuser_dnn_layer/user_dnn_layer/user_dnn_0/batch_normalization/gamma/part_0/Initializer/ones*
_output_shapes	
:*
T0*\
_classR
PNloc:@user_dnn_layer/user_dnn_layer/user_dnn_0/batch_normalization/gamma/part_0
Š
Nuser_dnn_layer/user_dnn_layer/user_dnn_0/batch_normalization/gamma/part_0/readIdentityIuser_dnn_layer/user_dnn_layer/user_dnn_0/batch_normalization/gamma/part_0*
T0*
_output_shapes	
:*\
_classR
PNloc:@user_dnn_layer/user_dnn_layer/user_dnn_0/batch_normalization/gamma/part_0

Zuser_dnn_layer/user_dnn_layer/user_dnn_0/batch_normalization/beta/part_0/Initializer/zerosConst*
_output_shapes	
:*
valueB*    *[
_classQ
OMloc:@user_dnn_layer/user_dnn_layer/user_dnn_0/batch_normalization/beta/part_0*
dtype0
ď
Huser_dnn_layer/user_dnn_layer/user_dnn_0/batch_normalization/beta/part_0
VariableV2*
dtype0*
shape:*
_output_shapes	
:*[
_classQ
OMloc:@user_dnn_layer/user_dnn_layer/user_dnn_0/batch_normalization/beta/part_0

Ouser_dnn_layer/user_dnn_layer/user_dnn_0/batch_normalization/beta/part_0/AssignAssignHuser_dnn_layer/user_dnn_layer/user_dnn_0/batch_normalization/beta/part_0Zuser_dnn_layer/user_dnn_layer/user_dnn_0/batch_normalization/beta/part_0/Initializer/zeros*
_output_shapes	
:*[
_classQ
OMloc:@user_dnn_layer/user_dnn_layer/user_dnn_0/batch_normalization/beta/part_0*
T0
Ś
Muser_dnn_layer/user_dnn_layer/user_dnn_0/batch_normalization/beta/part_0/readIdentityHuser_dnn_layer/user_dnn_layer/user_dnn_0/batch_normalization/beta/part_0*
T0*[
_classQ
OMloc:@user_dnn_layer/user_dnn_layer/user_dnn_0/batch_normalization/beta/part_0*
_output_shapes	
:

Zuser_dnn_layer/user_dnn_layer/user_dnn_0/batch_normalization/moving_mean/Initializer/zerosConst*[
_classQ
OMloc:@user_dnn_layer/user_dnn_layer/user_dnn_0/batch_normalization/moving_mean*
dtype0*
valueB*    *
_output_shapes	
:
ď
Huser_dnn_layer/user_dnn_layer/user_dnn_0/batch_normalization/moving_mean
VariableV2*
dtype0*
_output_shapes	
:*
shape:*[
_classQ
OMloc:@user_dnn_layer/user_dnn_layer/user_dnn_0/batch_normalization/moving_mean

Ouser_dnn_layer/user_dnn_layer/user_dnn_0/batch_normalization/moving_mean/AssignAssignHuser_dnn_layer/user_dnn_layer/user_dnn_0/batch_normalization/moving_meanZuser_dnn_layer/user_dnn_layer/user_dnn_0/batch_normalization/moving_mean/Initializer/zeros*[
_classQ
OMloc:@user_dnn_layer/user_dnn_layer/user_dnn_0/batch_normalization/moving_mean*
_output_shapes	
:*
T0
Ś
Muser_dnn_layer/user_dnn_layer/user_dnn_0/batch_normalization/moving_mean/readIdentityHuser_dnn_layer/user_dnn_layer/user_dnn_0/batch_normalization/moving_mean*
_output_shapes	
:*[
_classQ
OMloc:@user_dnn_layer/user_dnn_layer/user_dnn_0/batch_normalization/moving_mean*
T0

]user_dnn_layer/user_dnn_layer/user_dnn_0/batch_normalization/moving_variance/Initializer/onesConst*
_output_shapes	
:*
dtype0*_
_classU
SQloc:@user_dnn_layer/user_dnn_layer/user_dnn_0/batch_normalization/moving_variance*
valueB*  ?
÷
Luser_dnn_layer/user_dnn_layer/user_dnn_0/batch_normalization/moving_variance
VariableV2*
_output_shapes	
:*
dtype0*
shape:*_
_classU
SQloc:@user_dnn_layer/user_dnn_layer/user_dnn_0/batch_normalization/moving_variance

Suser_dnn_layer/user_dnn_layer/user_dnn_0/batch_normalization/moving_variance/AssignAssignLuser_dnn_layer/user_dnn_layer/user_dnn_0/batch_normalization/moving_variance]user_dnn_layer/user_dnn_layer/user_dnn_0/batch_normalization/moving_variance/Initializer/ones*
T0*_
_classU
SQloc:@user_dnn_layer/user_dnn_layer/user_dnn_0/batch_normalization/moving_variance*
_output_shapes	
:
˛
Quser_dnn_layer/user_dnn_layer/user_dnn_0/batch_normalization/moving_variance/readIdentityLuser_dnn_layer/user_dnn_layer/user_dnn_0/batch_normalization/moving_variance*
T0*_
_classU
SQloc:@user_dnn_layer/user_dnn_layer/user_dnn_0/batch_normalization/moving_variance*
_output_shapes	
:
Ľ
[user_dnn_layer/user_dnn_layer/user_dnn_0/batch_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
Ą
Iuser_dnn_layer/user_dnn_layer/user_dnn_0/batch_normalization/moments/meanMean6user_dnn_layer/user_dnn_layer/user_dnn_0/dense/BiasAdd[user_dnn_layer/user_dnn_layer/user_dnn_0/batch_normalization/moments/mean/reduction_indices*
	keep_dims(*
T0*
_output_shapes
:	
Ö
Quser_dnn_layer/user_dnn_layer/user_dnn_0/batch_normalization/moments/StopGradientStopGradientIuser_dnn_layer/user_dnn_layer/user_dnn_0/batch_normalization/moments/mean*
T0*
_output_shapes
:	
Š
Vuser_dnn_layer/user_dnn_layer/user_dnn_0/batch_normalization/moments/SquaredDifferenceSquaredDifference6user_dnn_layer/user_dnn_layer/user_dnn_0/dense/BiasAddQuser_dnn_layer/user_dnn_layer/user_dnn_0/batch_normalization/moments/StopGradient*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Š
_user_dnn_layer/user_dnn_layer/user_dnn_0/batch_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
valueB: *
dtype0
É
Muser_dnn_layer/user_dnn_layer/user_dnn_0/batch_normalization/moments/varianceMeanVuser_dnn_layer/user_dnn_layer/user_dnn_0/batch_normalization/moments/SquaredDifference_user_dnn_layer/user_dnn_layer/user_dnn_0/batch_normalization/moments/variance/reduction_indices*
	keep_dims(*
T0*
_output_shapes
:	
ß
Luser_dnn_layer/user_dnn_layer/user_dnn_0/batch_normalization/moments/SqueezeSqueezeIuser_dnn_layer/user_dnn_layer/user_dnn_0/batch_normalization/moments/mean*
_output_shapes	
:*
T0*
squeeze_dims
 
ĺ
Nuser_dnn_layer/user_dnn_layer/user_dnn_0/batch_normalization/moments/Squeeze_1SqueezeMuser_dnn_layer/user_dnn_layer/user_dnn_0/batch_normalization/moments/variance*
_output_shapes	
:*
squeeze_dims
 *
T0
ô
Ruser_dnn_layer/user_dnn_layer/user_dnn_0/batch_normalization/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<*[
_classQ
OMloc:@user_dnn_layer/user_dnn_layer/user_dnn_0/batch_normalization/moving_mean
÷
Puser_dnn_layer/user_dnn_layer/user_dnn_0/batch_normalization/AssignMovingAvg/subSubMuser_dnn_layer/user_dnn_layer/user_dnn_0/batch_normalization/moving_mean/readLuser_dnn_layer/user_dnn_layer/user_dnn_0/batch_normalization/moments/Squeeze*
T0*
_output_shapes	
:*[
_classQ
OMloc:@user_dnn_layer/user_dnn_layer/user_dnn_0/batch_normalization/moving_mean

Puser_dnn_layer/user_dnn_layer/user_dnn_0/batch_normalization/AssignMovingAvg/mulMulPuser_dnn_layer/user_dnn_layer/user_dnn_0/batch_normalization/AssignMovingAvg/subRuser_dnn_layer/user_dnn_layer/user_dnn_0/batch_normalization/AssignMovingAvg/decay*
T0*
_output_shapes	
:*[
_classQ
OMloc:@user_dnn_layer/user_dnn_layer/user_dnn_0/batch_normalization/moving_mean
ř
Luser_dnn_layer/user_dnn_layer/user_dnn_0/batch_normalization/AssignMovingAvg	AssignSubHuser_dnn_layer/user_dnn_layer/user_dnn_0/batch_normalization/moving_meanPuser_dnn_layer/user_dnn_layer/user_dnn_0/batch_normalization/AssignMovingAvg/mul*[
_classQ
OMloc:@user_dnn_layer/user_dnn_layer/user_dnn_0/batch_normalization/moving_mean*
_output_shapes	
:*
T0
ú
Tuser_dnn_layer/user_dnn_layer/user_dnn_0/batch_normalization/AssignMovingAvg_1/decayConst*
valueB
 *
×#<*_
_classU
SQloc:@user_dnn_layer/user_dnn_layer/user_dnn_0/batch_normalization/moving_variance*
dtype0*
_output_shapes
: 

Ruser_dnn_layer/user_dnn_layer/user_dnn_0/batch_normalization/AssignMovingAvg_1/subSubQuser_dnn_layer/user_dnn_layer/user_dnn_0/batch_normalization/moving_variance/readNuser_dnn_layer/user_dnn_layer/user_dnn_0/batch_normalization/moments/Squeeze_1*
T0*
_output_shapes	
:*_
_classU
SQloc:@user_dnn_layer/user_dnn_layer/user_dnn_0/batch_normalization/moving_variance

Ruser_dnn_layer/user_dnn_layer/user_dnn_0/batch_normalization/AssignMovingAvg_1/mulMulRuser_dnn_layer/user_dnn_layer/user_dnn_0/batch_normalization/AssignMovingAvg_1/subTuser_dnn_layer/user_dnn_layer/user_dnn_0/batch_normalization/AssignMovingAvg_1/decay*_
_classU
SQloc:@user_dnn_layer/user_dnn_layer/user_dnn_0/batch_normalization/moving_variance*
_output_shapes	
:*
T0

Nuser_dnn_layer/user_dnn_layer/user_dnn_0/batch_normalization/AssignMovingAvg_1	AssignSubLuser_dnn_layer/user_dnn_layer/user_dnn_0/batch_normalization/moving_varianceRuser_dnn_layer/user_dnn_layer/user_dnn_0/batch_normalization/AssignMovingAvg_1/mul*
_output_shapes	
:*
T0*_
_classU
SQloc:@user_dnn_layer/user_dnn_layer/user_dnn_0/batch_normalization/moving_variance
Â
Auser_dnn_layer/user_dnn_layer/user_dnn_0/batch_normalization/betaIdentityMuser_dnn_layer/user_dnn_layer/user_dnn_0/batch_normalization/beta/part_0/read*
_output_shapes	
:*
T0
Ä
Buser_dnn_layer/user_dnn_layer/user_dnn_0/batch_normalization/gammaIdentityNuser_dnn_layer/user_dnn_layer/user_dnn_0/batch_normalization/gamma/part_0/read*
_output_shapes	
:*
T0

Luser_dnn_layer/user_dnn_layer/user_dnn_0/batch_normalization/batchnorm/add/yConst*
valueB
 *o:*
dtype0*
_output_shapes
: 

Juser_dnn_layer/user_dnn_layer/user_dnn_0/batch_normalization/batchnorm/addAddV2Nuser_dnn_layer/user_dnn_layer/user_dnn_0/batch_normalization/moments/Squeeze_1Luser_dnn_layer/user_dnn_layer/user_dnn_0/batch_normalization/batchnorm/add/y*
_output_shapes	
:*
T0
Ç
Luser_dnn_layer/user_dnn_layer/user_dnn_0/batch_normalization/batchnorm/RsqrtRsqrtJuser_dnn_layer/user_dnn_layer/user_dnn_0/batch_normalization/batchnorm/add*
_output_shapes	
:*
T0

Juser_dnn_layer/user_dnn_layer/user_dnn_0/batch_normalization/batchnorm/mulMulLuser_dnn_layer/user_dnn_layer/user_dnn_0/batch_normalization/batchnorm/RsqrtBuser_dnn_layer/user_dnn_layer/user_dnn_0/batch_normalization/gamma*
_output_shapes	
:*
T0

Luser_dnn_layer/user_dnn_layer/user_dnn_0/batch_normalization/batchnorm/mul_1Mul6user_dnn_layer/user_dnn_layer/user_dnn_0/dense/BiasAddJuser_dnn_layer/user_dnn_layer/user_dnn_0/batch_normalization/batchnorm/mul*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

Luser_dnn_layer/user_dnn_layer/user_dnn_0/batch_normalization/batchnorm/mul_2MulLuser_dnn_layer/user_dnn_layer/user_dnn_0/batch_normalization/moments/SqueezeJuser_dnn_layer/user_dnn_layer/user_dnn_0/batch_normalization/batchnorm/mul*
T0*
_output_shapes	
:

Juser_dnn_layer/user_dnn_layer/user_dnn_0/batch_normalization/batchnorm/subSubAuser_dnn_layer/user_dnn_layer/user_dnn_0/batch_normalization/betaLuser_dnn_layer/user_dnn_layer/user_dnn_0/batch_normalization/batchnorm/mul_2*
T0*
_output_shapes	
:
˘
Luser_dnn_layer/user_dnn_layer/user_dnn_0/batch_normalization/batchnorm/add_1AddV2Luser_dnn_layer/user_dnn_layer/user_dnn_0/batch_normalization/batchnorm/mul_1Juser_dnn_layer/user_dnn_layer/user_dnn_0/batch_normalization/batchnorm/sub*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ś
-user_dnn_layer/user_dnn_layer/user_dnn_0/ReluReluLuser_dnn_layer/user_dnn_layer/user_dnn_0/batch_normalization/batchnorm/add_1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ł
;user_dnn_layer/user_dnn_layer/user_dnn_0/zero_fraction/SizeSize-user_dnn_layer/user_dnn_layer/user_dnn_0/Relu*
T0*
_output_shapes
: *
out_type0	

Buser_dnn_layer/user_dnn_layer/user_dnn_0/zero_fraction/LessEqual/yConst*
valueB	 R˙˙˙˙*
dtype0	*
_output_shapes
: 
ď
@user_dnn_layer/user_dnn_layer/user_dnn_0/zero_fraction/LessEqual	LessEqual;user_dnn_layer/user_dnn_layer/user_dnn_0/zero_fraction/SizeBuser_dnn_layer/user_dnn_layer/user_dnn_0/zero_fraction/LessEqual/y*
_output_shapes
: *
T0	
ó
Buser_dnn_layer/user_dnn_layer/user_dnn_0/zero_fraction/cond/SwitchSwitch@user_dnn_layer/user_dnn_layer/user_dnn_0/zero_fraction/LessEqual@user_dnn_layer/user_dnn_layer/user_dnn_0/zero_fraction/LessEqual*
_output_shapes
: : *
T0

ˇ
Duser_dnn_layer/user_dnn_layer/user_dnn_0/zero_fraction/cond/switch_tIdentityDuser_dnn_layer/user_dnn_layer/user_dnn_0/zero_fraction/cond/Switch:1*
T0
*
_output_shapes
: 
ľ
Duser_dnn_layer/user_dnn_layer/user_dnn_0/zero_fraction/cond/switch_fIdentityBuser_dnn_layer/user_dnn_layer/user_dnn_0/zero_fraction/cond/Switch*
_output_shapes
: *
T0

˛
Cuser_dnn_layer/user_dnn_layer/user_dnn_0/zero_fraction/cond/pred_idIdentity@user_dnn_layer/user_dnn_layer/user_dnn_0/zero_fraction/LessEqual*
_output_shapes
: *
T0

Ű
Ouser_dnn_layer/user_dnn_layer/user_dnn_0/zero_fraction/cond/count_nonzero/zerosConstE^user_dnn_layer/user_dnn_layer/user_dnn_0/zero_fraction/cond/switch_t*
valueB
 *    *
dtype0*
_output_shapes
: 
ż
Ruser_dnn_layer/user_dnn_layer/user_dnn_0/zero_fraction/cond/count_nonzero/NotEqualNotEqual[user_dnn_layer/user_dnn_layer/user_dnn_0/zero_fraction/cond/count_nonzero/NotEqual/Switch:1Ouser_dnn_layer/user_dnn_layer/user_dnn_0/zero_fraction/cond/count_nonzero/zeros*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
ŕ
Yuser_dnn_layer/user_dnn_layer/user_dnn_0/zero_fraction/cond/count_nonzero/NotEqual/SwitchSwitch-user_dnn_layer/user_dnn_layer/user_dnn_0/ReluCuser_dnn_layer/user_dnn_layer/user_dnn_0/zero_fraction/cond/pred_id*@
_class6
42loc:@user_dnn_layer/user_dnn_layer/user_dnn_0/Relu*<
_output_shapes*
(:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0
ě
Nuser_dnn_layer/user_dnn_layer/user_dnn_0/zero_fraction/cond/count_nonzero/CastCastRuser_dnn_layer/user_dnn_layer/user_dnn_0/zero_fraction/cond/count_nonzero/NotEqual*

DstT0*

SrcT0
*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ç
Ouser_dnn_layer/user_dnn_layer/user_dnn_0/zero_fraction/cond/count_nonzero/ConstConstE^user_dnn_layer/user_dnn_layer/user_dnn_0/zero_fraction/cond/switch_t*
dtype0*
_output_shapes
:*
valueB"       
 
Wuser_dnn_layer/user_dnn_layer/user_dnn_0/zero_fraction/cond/count_nonzero/nonzero_countSumNuser_dnn_layer/user_dnn_layer/user_dnn_0/zero_fraction/cond/count_nonzero/CastOuser_dnn_layer/user_dnn_layer/user_dnn_0/zero_fraction/cond/count_nonzero/Const*
_output_shapes
: *
T0
Ń
@user_dnn_layer/user_dnn_layer/user_dnn_0/zero_fraction/cond/CastCastWuser_dnn_layer/user_dnn_layer/user_dnn_0/zero_fraction/cond/count_nonzero/nonzero_count*

SrcT0*

DstT0	*
_output_shapes
: 
Ý
Quser_dnn_layer/user_dnn_layer/user_dnn_0/zero_fraction/cond/count_nonzero_1/zerosConstE^user_dnn_layer/user_dnn_layer/user_dnn_0/zero_fraction/cond/switch_f*
valueB
 *    *
dtype0*
_output_shapes
: 
Ă
Tuser_dnn_layer/user_dnn_layer/user_dnn_0/zero_fraction/cond/count_nonzero_1/NotEqualNotEqual[user_dnn_layer/user_dnn_layer/user_dnn_0/zero_fraction/cond/count_nonzero_1/NotEqual/SwitchQuser_dnn_layer/user_dnn_layer/user_dnn_0/zero_fraction/cond/count_nonzero_1/zeros*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
â
[user_dnn_layer/user_dnn_layer/user_dnn_0/zero_fraction/cond/count_nonzero_1/NotEqual/SwitchSwitch-user_dnn_layer/user_dnn_layer/user_dnn_0/ReluCuser_dnn_layer/user_dnn_layer/user_dnn_0/zero_fraction/cond/pred_id*<
_output_shapes*
(:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0*@
_class6
42loc:@user_dnn_layer/user_dnn_layer/user_dnn_0/Relu
đ
Puser_dnn_layer/user_dnn_layer/user_dnn_0/zero_fraction/cond/count_nonzero_1/CastCastTuser_dnn_layer/user_dnn_layer/user_dnn_0/zero_fraction/cond/count_nonzero_1/NotEqual*

DstT0	*

SrcT0
*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
é
Quser_dnn_layer/user_dnn_layer/user_dnn_0/zero_fraction/cond/count_nonzero_1/ConstConstE^user_dnn_layer/user_dnn_layer/user_dnn_0/zero_fraction/cond/switch_f*
_output_shapes
:*
valueB"       *
dtype0
Ś
Yuser_dnn_layer/user_dnn_layer/user_dnn_0/zero_fraction/cond/count_nonzero_1/nonzero_countSumPuser_dnn_layer/user_dnn_layer/user_dnn_0/zero_fraction/cond/count_nonzero_1/CastQuser_dnn_layer/user_dnn_layer/user_dnn_0/zero_fraction/cond/count_nonzero_1/Const*
T0	*
_output_shapes
: 

Auser_dnn_layer/user_dnn_layer/user_dnn_0/zero_fraction/cond/MergeMergeYuser_dnn_layer/user_dnn_layer/user_dnn_0/zero_fraction/cond/count_nonzero_1/nonzero_count@user_dnn_layer/user_dnn_layer/user_dnn_0/zero_fraction/cond/Cast*
N*
T0	*
_output_shapes
: : 
ő
Muser_dnn_layer/user_dnn_layer/user_dnn_0/zero_fraction/counts_to_fraction/subSub;user_dnn_layer/user_dnn_layer/user_dnn_0/zero_fraction/SizeAuser_dnn_layer/user_dnn_layer/user_dnn_0/zero_fraction/cond/Merge*
_output_shapes
: *
T0	
Ő
Nuser_dnn_layer/user_dnn_layer/user_dnn_0/zero_fraction/counts_to_fraction/CastCastMuser_dnn_layer/user_dnn_layer/user_dnn_0/zero_fraction/counts_to_fraction/sub*

SrcT0	*

DstT0*
_output_shapes
: 
Ĺ
Puser_dnn_layer/user_dnn_layer/user_dnn_0/zero_fraction/counts_to_fraction/Cast_1Cast;user_dnn_layer/user_dnn_layer/user_dnn_0/zero_fraction/Size*

SrcT0	*

DstT0*
_output_shapes
: 

Quser_dnn_layer/user_dnn_layer/user_dnn_0/zero_fraction/counts_to_fraction/truedivRealDivNuser_dnn_layer/user_dnn_layer/user_dnn_0/zero_fraction/counts_to_fraction/CastPuser_dnn_layer/user_dnn_layer/user_dnn_0/zero_fraction/counts_to_fraction/Cast_1*
_output_shapes
: *
T0
ż
?user_dnn_layer/user_dnn_layer/user_dnn_0/zero_fraction/fractionIdentityQuser_dnn_layer/user_dnn_layer/user_dnn_0/zero_fraction/counts_to_fraction/truediv*
T0*
_output_shapes
: 

nuser_dnn_layer/user_dnn_layer/user_dnn_0/user_dnn_layer/user_dnn_layer/user_dnn_0/fraction_of_zero_values/tagsConst*
_output_shapes
: *z
valueqBo Biuser_dnn_layer/user_dnn_layer/user_dnn_0/user_dnn_layer/user_dnn_layer/user_dnn_0/fraction_of_zero_values*
dtype0
Ě
iuser_dnn_layer/user_dnn_layer/user_dnn_0/user_dnn_layer/user_dnn_layer/user_dnn_0/fraction_of_zero_valuesScalarSummarynuser_dnn_layer/user_dnn_layer/user_dnn_0/user_dnn_layer/user_dnn_layer/user_dnn_0/fraction_of_zero_values/tags?user_dnn_layer/user_dnn_layer/user_dnn_0/zero_fraction/fraction*
T0*
_output_shapes
: 
ý
`user_dnn_layer/user_dnn_layer/user_dnn_0/user_dnn_layer/user_dnn_layer/user_dnn_0/activation/tagConst*
dtype0*
_output_shapes
: *m
valuedBb B\user_dnn_layer/user_dnn_layer/user_dnn_0/user_dnn_layer/user_dnn_layer/user_dnn_0/activation

\user_dnn_layer/user_dnn_layer/user_dnn_0/user_dnn_layer/user_dnn_layer/user_dnn_0/activationHistogramSummary`user_dnn_layer/user_dnn_layer/user_dnn_0/user_dnn_layer/user_dnn_layer/user_dnn_0/activation/tag-user_dnn_layer/user_dnn_layer/user_dnn_0/Relu*
_output_shapes
: 
˙
]user_dnn_layer/user_dnn_layer/user_dnn_1/dense/kernel/part_0/Initializer/random_uniform/shapeConst*
_output_shapes
:*O
_classE
CAloc:@user_dnn_layer/user_dnn_layer/user_dnn_1/dense/kernel/part_0*
dtype0*
valueB"      
ń
[user_dnn_layer/user_dnn_layer/user_dnn_1/dense/kernel/part_0/Initializer/random_uniform/minConst*
_output_shapes
: *O
_classE
CAloc:@user_dnn_layer/user_dnn_layer/user_dnn_1/dense/kernel/part_0*
valueB
 *   ž*
dtype0
ń
[user_dnn_layer/user_dnn_layer/user_dnn_1/dense/kernel/part_0/Initializer/random_uniform/maxConst*
_output_shapes
: *O
_classE
CAloc:@user_dnn_layer/user_dnn_layer/user_dnn_1/dense/kernel/part_0*
dtype0*
valueB
 *   >
Ţ
euser_dnn_layer/user_dnn_layer/user_dnn_1/dense/kernel/part_0/Initializer/random_uniform/RandomUniformRandomUniform]user_dnn_layer/user_dnn_layer/user_dnn_1/dense/kernel/part_0/Initializer/random_uniform/shape*O
_classE
CAloc:@user_dnn_layer/user_dnn_layer/user_dnn_1/dense/kernel/part_0*
T0*
dtype0* 
_output_shapes
:


[user_dnn_layer/user_dnn_layer/user_dnn_1/dense/kernel/part_0/Initializer/random_uniform/subSub[user_dnn_layer/user_dnn_layer/user_dnn_1/dense/kernel/part_0/Initializer/random_uniform/max[user_dnn_layer/user_dnn_layer/user_dnn_1/dense/kernel/part_0/Initializer/random_uniform/min*
T0*
_output_shapes
: *O
_classE
CAloc:@user_dnn_layer/user_dnn_layer/user_dnn_1/dense/kernel/part_0
˘
[user_dnn_layer/user_dnn_layer/user_dnn_1/dense/kernel/part_0/Initializer/random_uniform/mulMuleuser_dnn_layer/user_dnn_layer/user_dnn_1/dense/kernel/part_0/Initializer/random_uniform/RandomUniform[user_dnn_layer/user_dnn_layer/user_dnn_1/dense/kernel/part_0/Initializer/random_uniform/sub*
T0* 
_output_shapes
:
*O
_classE
CAloc:@user_dnn_layer/user_dnn_layer/user_dnn_1/dense/kernel/part_0

Wuser_dnn_layer/user_dnn_layer/user_dnn_1/dense/kernel/part_0/Initializer/random_uniformAdd[user_dnn_layer/user_dnn_layer/user_dnn_1/dense/kernel/part_0/Initializer/random_uniform/mul[user_dnn_layer/user_dnn_layer/user_dnn_1/dense/kernel/part_0/Initializer/random_uniform/min*O
_classE
CAloc:@user_dnn_layer/user_dnn_layer/user_dnn_1/dense/kernel/part_0*
T0* 
_output_shapes
:

á
<user_dnn_layer/user_dnn_layer/user_dnn_1/dense/kernel/part_0
VariableV2*
dtype0* 
_output_shapes
:
*
shape:
*O
_classE
CAloc:@user_dnn_layer/user_dnn_layer/user_dnn_1/dense/kernel/part_0
ŕ
Cuser_dnn_layer/user_dnn_layer/user_dnn_1/dense/kernel/part_0/AssignAssign<user_dnn_layer/user_dnn_layer/user_dnn_1/dense/kernel/part_0Wuser_dnn_layer/user_dnn_layer/user_dnn_1/dense/kernel/part_0/Initializer/random_uniform* 
_output_shapes
:
*
T0*O
_classE
CAloc:@user_dnn_layer/user_dnn_layer/user_dnn_1/dense/kernel/part_0

Auser_dnn_layer/user_dnn_layer/user_dnn_1/dense/kernel/part_0/readIdentity<user_dnn_layer/user_dnn_layer/user_dnn_1/dense/kernel/part_0*
T0*O
_classE
CAloc:@user_dnn_layer/user_dnn_layer/user_dnn_1/dense/kernel/part_0* 
_output_shapes
:

ę
Luser_dnn_layer/user_dnn_layer/user_dnn_1/dense/bias/part_0/Initializer/zerosConst*
dtype0*
valueB*    *
_output_shapes	
:*M
_classC
A?loc:@user_dnn_layer/user_dnn_layer/user_dnn_1/dense/bias/part_0
Ó
:user_dnn_layer/user_dnn_layer/user_dnn_1/dense/bias/part_0
VariableV2*
shape:*M
_classC
A?loc:@user_dnn_layer/user_dnn_layer/user_dnn_1/dense/bias/part_0*
dtype0*
_output_shapes	
:
Ę
Auser_dnn_layer/user_dnn_layer/user_dnn_1/dense/bias/part_0/AssignAssign:user_dnn_layer/user_dnn_layer/user_dnn_1/dense/bias/part_0Luser_dnn_layer/user_dnn_layer/user_dnn_1/dense/bias/part_0/Initializer/zeros*
_output_shapes	
:*M
_classC
A?loc:@user_dnn_layer/user_dnn_layer/user_dnn_1/dense/bias/part_0*
T0
ü
?user_dnn_layer/user_dnn_layer/user_dnn_1/dense/bias/part_0/readIdentity:user_dnn_layer/user_dnn_layer/user_dnn_1/dense/bias/part_0*
_output_shapes	
:*M
_classC
A?loc:@user_dnn_layer/user_dnn_layer/user_dnn_1/dense/bias/part_0*
T0
Ż
5user_dnn_layer/user_dnn_layer/user_dnn_1/dense/kernelIdentityAuser_dnn_layer/user_dnn_layer/user_dnn_1/dense/kernel/part_0/read*
T0* 
_output_shapes
:

Ř
5user_dnn_layer/user_dnn_layer/user_dnn_1/dense/MatMulMatMul-user_dnn_layer/user_dnn_layer/user_dnn_0/Relu5user_dnn_layer/user_dnn_layer/user_dnn_1/dense/kernel*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ś
3user_dnn_layer/user_dnn_layer/user_dnn_1/dense/biasIdentity?user_dnn_layer/user_dnn_layer/user_dnn_1/dense/bias/part_0/read*
_output_shapes	
:*
T0
ŕ
6user_dnn_layer/user_dnn_layer/user_dnn_1/dense/BiasAddBiasAdd5user_dnn_layer/user_dnn_layer/user_dnn_1/dense/MatMul3user_dnn_layer/user_dnn_layer/user_dnn_1/dense/bias*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

Zuser_dnn_layer/user_dnn_layer/user_dnn_1/batch_normalization/gamma/part_0/Initializer/onesConst*\
_classR
PNloc:@user_dnn_layer/user_dnn_layer/user_dnn_1/batch_normalization/gamma/part_0*
dtype0*
valueB*  ?*
_output_shapes	
:
ń
Iuser_dnn_layer/user_dnn_layer/user_dnn_1/batch_normalization/gamma/part_0
VariableV2*\
_classR
PNloc:@user_dnn_layer/user_dnn_layer/user_dnn_1/batch_normalization/gamma/part_0*
shape:*
dtype0*
_output_shapes	
:

Puser_dnn_layer/user_dnn_layer/user_dnn_1/batch_normalization/gamma/part_0/AssignAssignIuser_dnn_layer/user_dnn_layer/user_dnn_1/batch_normalization/gamma/part_0Zuser_dnn_layer/user_dnn_layer/user_dnn_1/batch_normalization/gamma/part_0/Initializer/ones*
T0*
_output_shapes	
:*\
_classR
PNloc:@user_dnn_layer/user_dnn_layer/user_dnn_1/batch_normalization/gamma/part_0
Š
Nuser_dnn_layer/user_dnn_layer/user_dnn_1/batch_normalization/gamma/part_0/readIdentityIuser_dnn_layer/user_dnn_layer/user_dnn_1/batch_normalization/gamma/part_0*
T0*
_output_shapes	
:*\
_classR
PNloc:@user_dnn_layer/user_dnn_layer/user_dnn_1/batch_normalization/gamma/part_0

Zuser_dnn_layer/user_dnn_layer/user_dnn_1/batch_normalization/beta/part_0/Initializer/zerosConst*
valueB*    *
dtype0*[
_classQ
OMloc:@user_dnn_layer/user_dnn_layer/user_dnn_1/batch_normalization/beta/part_0*
_output_shapes	
:
ď
Huser_dnn_layer/user_dnn_layer/user_dnn_1/batch_normalization/beta/part_0
VariableV2*[
_classQ
OMloc:@user_dnn_layer/user_dnn_layer/user_dnn_1/batch_normalization/beta/part_0*
shape:*
_output_shapes	
:*
dtype0

Ouser_dnn_layer/user_dnn_layer/user_dnn_1/batch_normalization/beta/part_0/AssignAssignHuser_dnn_layer/user_dnn_layer/user_dnn_1/batch_normalization/beta/part_0Zuser_dnn_layer/user_dnn_layer/user_dnn_1/batch_normalization/beta/part_0/Initializer/zeros*[
_classQ
OMloc:@user_dnn_layer/user_dnn_layer/user_dnn_1/batch_normalization/beta/part_0*
T0*
_output_shapes	
:
Ś
Muser_dnn_layer/user_dnn_layer/user_dnn_1/batch_normalization/beta/part_0/readIdentityHuser_dnn_layer/user_dnn_layer/user_dnn_1/batch_normalization/beta/part_0*
T0*[
_classQ
OMloc:@user_dnn_layer/user_dnn_layer/user_dnn_1/batch_normalization/beta/part_0*
_output_shapes	
:

Zuser_dnn_layer/user_dnn_layer/user_dnn_1/batch_normalization/moving_mean/Initializer/zerosConst*
valueB*    *[
_classQ
OMloc:@user_dnn_layer/user_dnn_layer/user_dnn_1/batch_normalization/moving_mean*
dtype0*
_output_shapes	
:
ď
Huser_dnn_layer/user_dnn_layer/user_dnn_1/batch_normalization/moving_mean
VariableV2*
dtype0*
shape:*
_output_shapes	
:*[
_classQ
OMloc:@user_dnn_layer/user_dnn_layer/user_dnn_1/batch_normalization/moving_mean

Ouser_dnn_layer/user_dnn_layer/user_dnn_1/batch_normalization/moving_mean/AssignAssignHuser_dnn_layer/user_dnn_layer/user_dnn_1/batch_normalization/moving_meanZuser_dnn_layer/user_dnn_layer/user_dnn_1/batch_normalization/moving_mean/Initializer/zeros*
T0*
_output_shapes	
:*[
_classQ
OMloc:@user_dnn_layer/user_dnn_layer/user_dnn_1/batch_normalization/moving_mean
Ś
Muser_dnn_layer/user_dnn_layer/user_dnn_1/batch_normalization/moving_mean/readIdentityHuser_dnn_layer/user_dnn_layer/user_dnn_1/batch_normalization/moving_mean*
T0*
_output_shapes	
:*[
_classQ
OMloc:@user_dnn_layer/user_dnn_layer/user_dnn_1/batch_normalization/moving_mean

]user_dnn_layer/user_dnn_layer/user_dnn_1/batch_normalization/moving_variance/Initializer/onesConst*_
_classU
SQloc:@user_dnn_layer/user_dnn_layer/user_dnn_1/batch_normalization/moving_variance*
_output_shapes	
:*
dtype0*
valueB*  ?
÷
Luser_dnn_layer/user_dnn_layer/user_dnn_1/batch_normalization/moving_variance
VariableV2*
_output_shapes	
:*_
_classU
SQloc:@user_dnn_layer/user_dnn_layer/user_dnn_1/batch_normalization/moving_variance*
dtype0*
shape:

Suser_dnn_layer/user_dnn_layer/user_dnn_1/batch_normalization/moving_variance/AssignAssignLuser_dnn_layer/user_dnn_layer/user_dnn_1/batch_normalization/moving_variance]user_dnn_layer/user_dnn_layer/user_dnn_1/batch_normalization/moving_variance/Initializer/ones*_
_classU
SQloc:@user_dnn_layer/user_dnn_layer/user_dnn_1/batch_normalization/moving_variance*
_output_shapes	
:*
T0
˛
Quser_dnn_layer/user_dnn_layer/user_dnn_1/batch_normalization/moving_variance/readIdentityLuser_dnn_layer/user_dnn_layer/user_dnn_1/batch_normalization/moving_variance*_
_classU
SQloc:@user_dnn_layer/user_dnn_layer/user_dnn_1/batch_normalization/moving_variance*
_output_shapes	
:*
T0
Ľ
[user_dnn_layer/user_dnn_layer/user_dnn_1/batch_normalization/moments/mean/reduction_indicesConst*
dtype0*
_output_shapes
:*
valueB: 
Ą
Iuser_dnn_layer/user_dnn_layer/user_dnn_1/batch_normalization/moments/meanMean6user_dnn_layer/user_dnn_layer/user_dnn_1/dense/BiasAdd[user_dnn_layer/user_dnn_layer/user_dnn_1/batch_normalization/moments/mean/reduction_indices*
	keep_dims(*
_output_shapes
:	*
T0
Ö
Quser_dnn_layer/user_dnn_layer/user_dnn_1/batch_normalization/moments/StopGradientStopGradientIuser_dnn_layer/user_dnn_layer/user_dnn_1/batch_normalization/moments/mean*
_output_shapes
:	*
T0
Š
Vuser_dnn_layer/user_dnn_layer/user_dnn_1/batch_normalization/moments/SquaredDifferenceSquaredDifference6user_dnn_layer/user_dnn_layer/user_dnn_1/dense/BiasAddQuser_dnn_layer/user_dnn_layer/user_dnn_1/batch_normalization/moments/StopGradient*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Š
_user_dnn_layer/user_dnn_layer/user_dnn_1/batch_normalization/moments/variance/reduction_indicesConst*
dtype0*
_output_shapes
:*
valueB: 
É
Muser_dnn_layer/user_dnn_layer/user_dnn_1/batch_normalization/moments/varianceMeanVuser_dnn_layer/user_dnn_layer/user_dnn_1/batch_normalization/moments/SquaredDifference_user_dnn_layer/user_dnn_layer/user_dnn_1/batch_normalization/moments/variance/reduction_indices*
	keep_dims(*
_output_shapes
:	*
T0
ß
Luser_dnn_layer/user_dnn_layer/user_dnn_1/batch_normalization/moments/SqueezeSqueezeIuser_dnn_layer/user_dnn_layer/user_dnn_1/batch_normalization/moments/mean*
_output_shapes	
:*
T0*
squeeze_dims
 
ĺ
Nuser_dnn_layer/user_dnn_layer/user_dnn_1/batch_normalization/moments/Squeeze_1SqueezeMuser_dnn_layer/user_dnn_layer/user_dnn_1/batch_normalization/moments/variance*
squeeze_dims
 *
T0*
_output_shapes	
:
ô
Ruser_dnn_layer/user_dnn_layer/user_dnn_1/batch_normalization/AssignMovingAvg/decayConst*
dtype0*[
_classQ
OMloc:@user_dnn_layer/user_dnn_layer/user_dnn_1/batch_normalization/moving_mean*
valueB
 *
×#<*
_output_shapes
: 
÷
Puser_dnn_layer/user_dnn_layer/user_dnn_1/batch_normalization/AssignMovingAvg/subSubMuser_dnn_layer/user_dnn_layer/user_dnn_1/batch_normalization/moving_mean/readLuser_dnn_layer/user_dnn_layer/user_dnn_1/batch_normalization/moments/Squeeze*
T0*[
_classQ
OMloc:@user_dnn_layer/user_dnn_layer/user_dnn_1/batch_normalization/moving_mean*
_output_shapes	
:

Puser_dnn_layer/user_dnn_layer/user_dnn_1/batch_normalization/AssignMovingAvg/mulMulPuser_dnn_layer/user_dnn_layer/user_dnn_1/batch_normalization/AssignMovingAvg/subRuser_dnn_layer/user_dnn_layer/user_dnn_1/batch_normalization/AssignMovingAvg/decay*[
_classQ
OMloc:@user_dnn_layer/user_dnn_layer/user_dnn_1/batch_normalization/moving_mean*
T0*
_output_shapes	
:
ř
Luser_dnn_layer/user_dnn_layer/user_dnn_1/batch_normalization/AssignMovingAvg	AssignSubHuser_dnn_layer/user_dnn_layer/user_dnn_1/batch_normalization/moving_meanPuser_dnn_layer/user_dnn_layer/user_dnn_1/batch_normalization/AssignMovingAvg/mul*[
_classQ
OMloc:@user_dnn_layer/user_dnn_layer/user_dnn_1/batch_normalization/moving_mean*
_output_shapes	
:*
T0
ú
Tuser_dnn_layer/user_dnn_layer/user_dnn_1/batch_normalization/AssignMovingAvg_1/decayConst*
_output_shapes
: *_
_classU
SQloc:@user_dnn_layer/user_dnn_layer/user_dnn_1/batch_normalization/moving_variance*
dtype0*
valueB
 *
×#<

Ruser_dnn_layer/user_dnn_layer/user_dnn_1/batch_normalization/AssignMovingAvg_1/subSubQuser_dnn_layer/user_dnn_layer/user_dnn_1/batch_normalization/moving_variance/readNuser_dnn_layer/user_dnn_layer/user_dnn_1/batch_normalization/moments/Squeeze_1*
T0*_
_classU
SQloc:@user_dnn_layer/user_dnn_layer/user_dnn_1/batch_normalization/moving_variance*
_output_shapes	
:

Ruser_dnn_layer/user_dnn_layer/user_dnn_1/batch_normalization/AssignMovingAvg_1/mulMulRuser_dnn_layer/user_dnn_layer/user_dnn_1/batch_normalization/AssignMovingAvg_1/subTuser_dnn_layer/user_dnn_layer/user_dnn_1/batch_normalization/AssignMovingAvg_1/decay*
T0*
_output_shapes	
:*_
_classU
SQloc:@user_dnn_layer/user_dnn_layer/user_dnn_1/batch_normalization/moving_variance

Nuser_dnn_layer/user_dnn_layer/user_dnn_1/batch_normalization/AssignMovingAvg_1	AssignSubLuser_dnn_layer/user_dnn_layer/user_dnn_1/batch_normalization/moving_varianceRuser_dnn_layer/user_dnn_layer/user_dnn_1/batch_normalization/AssignMovingAvg_1/mul*
T0*_
_classU
SQloc:@user_dnn_layer/user_dnn_layer/user_dnn_1/batch_normalization/moving_variance*
_output_shapes	
:
Â
Auser_dnn_layer/user_dnn_layer/user_dnn_1/batch_normalization/betaIdentityMuser_dnn_layer/user_dnn_layer/user_dnn_1/batch_normalization/beta/part_0/read*
T0*
_output_shapes	
:
Ä
Buser_dnn_layer/user_dnn_layer/user_dnn_1/batch_normalization/gammaIdentityNuser_dnn_layer/user_dnn_layer/user_dnn_1/batch_normalization/gamma/part_0/read*
_output_shapes	
:*
T0

Luser_dnn_layer/user_dnn_layer/user_dnn_1/batch_normalization/batchnorm/add/yConst*
valueB
 *o:*
_output_shapes
: *
dtype0

Juser_dnn_layer/user_dnn_layer/user_dnn_1/batch_normalization/batchnorm/addAddV2Nuser_dnn_layer/user_dnn_layer/user_dnn_1/batch_normalization/moments/Squeeze_1Luser_dnn_layer/user_dnn_layer/user_dnn_1/batch_normalization/batchnorm/add/y*
_output_shapes	
:*
T0
Ç
Luser_dnn_layer/user_dnn_layer/user_dnn_1/batch_normalization/batchnorm/RsqrtRsqrtJuser_dnn_layer/user_dnn_layer/user_dnn_1/batch_normalization/batchnorm/add*
_output_shapes	
:*
T0

Juser_dnn_layer/user_dnn_layer/user_dnn_1/batch_normalization/batchnorm/mulMulLuser_dnn_layer/user_dnn_layer/user_dnn_1/batch_normalization/batchnorm/RsqrtBuser_dnn_layer/user_dnn_layer/user_dnn_1/batch_normalization/gamma*
_output_shapes	
:*
T0

Luser_dnn_layer/user_dnn_layer/user_dnn_1/batch_normalization/batchnorm/mul_1Mul6user_dnn_layer/user_dnn_layer/user_dnn_1/dense/BiasAddJuser_dnn_layer/user_dnn_layer/user_dnn_1/batch_normalization/batchnorm/mul*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

Luser_dnn_layer/user_dnn_layer/user_dnn_1/batch_normalization/batchnorm/mul_2MulLuser_dnn_layer/user_dnn_layer/user_dnn_1/batch_normalization/moments/SqueezeJuser_dnn_layer/user_dnn_layer/user_dnn_1/batch_normalization/batchnorm/mul*
T0*
_output_shapes	
:

Juser_dnn_layer/user_dnn_layer/user_dnn_1/batch_normalization/batchnorm/subSubAuser_dnn_layer/user_dnn_layer/user_dnn_1/batch_normalization/betaLuser_dnn_layer/user_dnn_layer/user_dnn_1/batch_normalization/batchnorm/mul_2*
_output_shapes	
:*
T0
˘
Luser_dnn_layer/user_dnn_layer/user_dnn_1/batch_normalization/batchnorm/add_1AddV2Luser_dnn_layer/user_dnn_layer/user_dnn_1/batch_normalization/batchnorm/mul_1Juser_dnn_layer/user_dnn_layer/user_dnn_1/batch_normalization/batchnorm/sub*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
ś
-user_dnn_layer/user_dnn_layer/user_dnn_1/ReluReluLuser_dnn_layer/user_dnn_layer/user_dnn_1/batch_normalization/batchnorm/add_1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ł
;user_dnn_layer/user_dnn_layer/user_dnn_1/zero_fraction/SizeSize-user_dnn_layer/user_dnn_layer/user_dnn_1/Relu*
_output_shapes
: *
out_type0	*
T0

Buser_dnn_layer/user_dnn_layer/user_dnn_1/zero_fraction/LessEqual/yConst*
_output_shapes
: *
dtype0	*
valueB	 R˙˙˙˙
ď
@user_dnn_layer/user_dnn_layer/user_dnn_1/zero_fraction/LessEqual	LessEqual;user_dnn_layer/user_dnn_layer/user_dnn_1/zero_fraction/SizeBuser_dnn_layer/user_dnn_layer/user_dnn_1/zero_fraction/LessEqual/y*
_output_shapes
: *
T0	
ó
Buser_dnn_layer/user_dnn_layer/user_dnn_1/zero_fraction/cond/SwitchSwitch@user_dnn_layer/user_dnn_layer/user_dnn_1/zero_fraction/LessEqual@user_dnn_layer/user_dnn_layer/user_dnn_1/zero_fraction/LessEqual*
_output_shapes
: : *
T0

ˇ
Duser_dnn_layer/user_dnn_layer/user_dnn_1/zero_fraction/cond/switch_tIdentityDuser_dnn_layer/user_dnn_layer/user_dnn_1/zero_fraction/cond/Switch:1*
T0
*
_output_shapes
: 
ľ
Duser_dnn_layer/user_dnn_layer/user_dnn_1/zero_fraction/cond/switch_fIdentityBuser_dnn_layer/user_dnn_layer/user_dnn_1/zero_fraction/cond/Switch*
_output_shapes
: *
T0

˛
Cuser_dnn_layer/user_dnn_layer/user_dnn_1/zero_fraction/cond/pred_idIdentity@user_dnn_layer/user_dnn_layer/user_dnn_1/zero_fraction/LessEqual*
T0
*
_output_shapes
: 
Ű
Ouser_dnn_layer/user_dnn_layer/user_dnn_1/zero_fraction/cond/count_nonzero/zerosConstE^user_dnn_layer/user_dnn_layer/user_dnn_1/zero_fraction/cond/switch_t*
dtype0*
valueB
 *    *
_output_shapes
: 
ż
Ruser_dnn_layer/user_dnn_layer/user_dnn_1/zero_fraction/cond/count_nonzero/NotEqualNotEqual[user_dnn_layer/user_dnn_layer/user_dnn_1/zero_fraction/cond/count_nonzero/NotEqual/Switch:1Ouser_dnn_layer/user_dnn_layer/user_dnn_1/zero_fraction/cond/count_nonzero/zeros*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
ŕ
Yuser_dnn_layer/user_dnn_layer/user_dnn_1/zero_fraction/cond/count_nonzero/NotEqual/SwitchSwitch-user_dnn_layer/user_dnn_layer/user_dnn_1/ReluCuser_dnn_layer/user_dnn_layer/user_dnn_1/zero_fraction/cond/pred_id*<
_output_shapes*
(:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0*@
_class6
42loc:@user_dnn_layer/user_dnn_layer/user_dnn_1/Relu
ě
Nuser_dnn_layer/user_dnn_layer/user_dnn_1/zero_fraction/cond/count_nonzero/CastCastRuser_dnn_layer/user_dnn_layer/user_dnn_1/zero_fraction/cond/count_nonzero/NotEqual*

DstT0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*

SrcT0

ç
Ouser_dnn_layer/user_dnn_layer/user_dnn_1/zero_fraction/cond/count_nonzero/ConstConstE^user_dnn_layer/user_dnn_layer/user_dnn_1/zero_fraction/cond/switch_t*
dtype0*
_output_shapes
:*
valueB"       
 
Wuser_dnn_layer/user_dnn_layer/user_dnn_1/zero_fraction/cond/count_nonzero/nonzero_countSumNuser_dnn_layer/user_dnn_layer/user_dnn_1/zero_fraction/cond/count_nonzero/CastOuser_dnn_layer/user_dnn_layer/user_dnn_1/zero_fraction/cond/count_nonzero/Const*
T0*
_output_shapes
: 
Ń
@user_dnn_layer/user_dnn_layer/user_dnn_1/zero_fraction/cond/CastCastWuser_dnn_layer/user_dnn_layer/user_dnn_1/zero_fraction/cond/count_nonzero/nonzero_count*

DstT0	*
_output_shapes
: *

SrcT0
Ý
Quser_dnn_layer/user_dnn_layer/user_dnn_1/zero_fraction/cond/count_nonzero_1/zerosConstE^user_dnn_layer/user_dnn_layer/user_dnn_1/zero_fraction/cond/switch_f*
dtype0*
_output_shapes
: *
valueB
 *    
Ă
Tuser_dnn_layer/user_dnn_layer/user_dnn_1/zero_fraction/cond/count_nonzero_1/NotEqualNotEqual[user_dnn_layer/user_dnn_layer/user_dnn_1/zero_fraction/cond/count_nonzero_1/NotEqual/SwitchQuser_dnn_layer/user_dnn_layer/user_dnn_1/zero_fraction/cond/count_nonzero_1/zeros*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
â
[user_dnn_layer/user_dnn_layer/user_dnn_1/zero_fraction/cond/count_nonzero_1/NotEqual/SwitchSwitch-user_dnn_layer/user_dnn_layer/user_dnn_1/ReluCuser_dnn_layer/user_dnn_layer/user_dnn_1/zero_fraction/cond/pred_id*<
_output_shapes*
(:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*@
_class6
42loc:@user_dnn_layer/user_dnn_layer/user_dnn_1/Relu*
T0
đ
Puser_dnn_layer/user_dnn_layer/user_dnn_1/zero_fraction/cond/count_nonzero_1/CastCastTuser_dnn_layer/user_dnn_layer/user_dnn_1/zero_fraction/cond/count_nonzero_1/NotEqual*

SrcT0
*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*

DstT0	
é
Quser_dnn_layer/user_dnn_layer/user_dnn_1/zero_fraction/cond/count_nonzero_1/ConstConstE^user_dnn_layer/user_dnn_layer/user_dnn_1/zero_fraction/cond/switch_f*
dtype0*
valueB"       *
_output_shapes
:
Ś
Yuser_dnn_layer/user_dnn_layer/user_dnn_1/zero_fraction/cond/count_nonzero_1/nonzero_countSumPuser_dnn_layer/user_dnn_layer/user_dnn_1/zero_fraction/cond/count_nonzero_1/CastQuser_dnn_layer/user_dnn_layer/user_dnn_1/zero_fraction/cond/count_nonzero_1/Const*
T0	*
_output_shapes
: 

Auser_dnn_layer/user_dnn_layer/user_dnn_1/zero_fraction/cond/MergeMergeYuser_dnn_layer/user_dnn_layer/user_dnn_1/zero_fraction/cond/count_nonzero_1/nonzero_count@user_dnn_layer/user_dnn_layer/user_dnn_1/zero_fraction/cond/Cast*
T0	*
N*
_output_shapes
: : 
ő
Muser_dnn_layer/user_dnn_layer/user_dnn_1/zero_fraction/counts_to_fraction/subSub;user_dnn_layer/user_dnn_layer/user_dnn_1/zero_fraction/SizeAuser_dnn_layer/user_dnn_layer/user_dnn_1/zero_fraction/cond/Merge*
T0	*
_output_shapes
: 
Ő
Nuser_dnn_layer/user_dnn_layer/user_dnn_1/zero_fraction/counts_to_fraction/CastCastMuser_dnn_layer/user_dnn_layer/user_dnn_1/zero_fraction/counts_to_fraction/sub*

SrcT0	*

DstT0*
_output_shapes
: 
Ĺ
Puser_dnn_layer/user_dnn_layer/user_dnn_1/zero_fraction/counts_to_fraction/Cast_1Cast;user_dnn_layer/user_dnn_layer/user_dnn_1/zero_fraction/Size*
_output_shapes
: *

SrcT0	*

DstT0

Quser_dnn_layer/user_dnn_layer/user_dnn_1/zero_fraction/counts_to_fraction/truedivRealDivNuser_dnn_layer/user_dnn_layer/user_dnn_1/zero_fraction/counts_to_fraction/CastPuser_dnn_layer/user_dnn_layer/user_dnn_1/zero_fraction/counts_to_fraction/Cast_1*
T0*
_output_shapes
: 
ż
?user_dnn_layer/user_dnn_layer/user_dnn_1/zero_fraction/fractionIdentityQuser_dnn_layer/user_dnn_layer/user_dnn_1/zero_fraction/counts_to_fraction/truediv*
_output_shapes
: *
T0

nuser_dnn_layer/user_dnn_layer/user_dnn_1/user_dnn_layer/user_dnn_layer/user_dnn_1/fraction_of_zero_values/tagsConst*
_output_shapes
: *z
valueqBo Biuser_dnn_layer/user_dnn_layer/user_dnn_1/user_dnn_layer/user_dnn_layer/user_dnn_1/fraction_of_zero_values*
dtype0
Ě
iuser_dnn_layer/user_dnn_layer/user_dnn_1/user_dnn_layer/user_dnn_layer/user_dnn_1/fraction_of_zero_valuesScalarSummarynuser_dnn_layer/user_dnn_layer/user_dnn_1/user_dnn_layer/user_dnn_layer/user_dnn_1/fraction_of_zero_values/tags?user_dnn_layer/user_dnn_layer/user_dnn_1/zero_fraction/fraction*
_output_shapes
: *
T0
ý
`user_dnn_layer/user_dnn_layer/user_dnn_1/user_dnn_layer/user_dnn_layer/user_dnn_1/activation/tagConst*m
valuedBb B\user_dnn_layer/user_dnn_layer/user_dnn_1/user_dnn_layer/user_dnn_layer/user_dnn_1/activation*
_output_shapes
: *
dtype0

\user_dnn_layer/user_dnn_layer/user_dnn_1/user_dnn_layer/user_dnn_layer/user_dnn_1/activationHistogramSummary`user_dnn_layer/user_dnn_layer/user_dnn_1/user_dnn_layer/user_dnn_layer/user_dnn_1/activation/tag-user_dnn_layer/user_dnn_layer/user_dnn_1/Relu*
_output_shapes
: 
˙
]user_dnn_layer/user_dnn_layer/user_dnn_2/dense/kernel/part_0/Initializer/random_uniform/shapeConst*
valueB"   @   *
dtype0*O
_classE
CAloc:@user_dnn_layer/user_dnn_layer/user_dnn_2/dense/kernel/part_0*
_output_shapes
:
ń
[user_dnn_layer/user_dnn_layer/user_dnn_2/dense/kernel/part_0/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *O
_classE
CAloc:@user_dnn_layer/user_dnn_layer/user_dnn_2/dense/kernel/part_0*
valueB
 *ó5ž
ń
[user_dnn_layer/user_dnn_layer/user_dnn_2/dense/kernel/part_0/Initializer/random_uniform/maxConst*O
_classE
CAloc:@user_dnn_layer/user_dnn_layer/user_dnn_2/dense/kernel/part_0*
dtype0*
_output_shapes
: *
valueB
 *ó5>
Ý
euser_dnn_layer/user_dnn_layer/user_dnn_2/dense/kernel/part_0/Initializer/random_uniform/RandomUniformRandomUniform]user_dnn_layer/user_dnn_layer/user_dnn_2/dense/kernel/part_0/Initializer/random_uniform/shape*O
_classE
CAloc:@user_dnn_layer/user_dnn_layer/user_dnn_2/dense/kernel/part_0*
T0*
_output_shapes
:	@*
dtype0

[user_dnn_layer/user_dnn_layer/user_dnn_2/dense/kernel/part_0/Initializer/random_uniform/subSub[user_dnn_layer/user_dnn_layer/user_dnn_2/dense/kernel/part_0/Initializer/random_uniform/max[user_dnn_layer/user_dnn_layer/user_dnn_2/dense/kernel/part_0/Initializer/random_uniform/min*
_output_shapes
: *
T0*O
_classE
CAloc:@user_dnn_layer/user_dnn_layer/user_dnn_2/dense/kernel/part_0
Ą
[user_dnn_layer/user_dnn_layer/user_dnn_2/dense/kernel/part_0/Initializer/random_uniform/mulMuleuser_dnn_layer/user_dnn_layer/user_dnn_2/dense/kernel/part_0/Initializer/random_uniform/RandomUniform[user_dnn_layer/user_dnn_layer/user_dnn_2/dense/kernel/part_0/Initializer/random_uniform/sub*
T0*O
_classE
CAloc:@user_dnn_layer/user_dnn_layer/user_dnn_2/dense/kernel/part_0*
_output_shapes
:	@

Wuser_dnn_layer/user_dnn_layer/user_dnn_2/dense/kernel/part_0/Initializer/random_uniformAdd[user_dnn_layer/user_dnn_layer/user_dnn_2/dense/kernel/part_0/Initializer/random_uniform/mul[user_dnn_layer/user_dnn_layer/user_dnn_2/dense/kernel/part_0/Initializer/random_uniform/min*
T0*
_output_shapes
:	@*O
_classE
CAloc:@user_dnn_layer/user_dnn_layer/user_dnn_2/dense/kernel/part_0
ß
<user_dnn_layer/user_dnn_layer/user_dnn_2/dense/kernel/part_0
VariableV2*
_output_shapes
:	@*
dtype0*
shape:	@*O
_classE
CAloc:@user_dnn_layer/user_dnn_layer/user_dnn_2/dense/kernel/part_0
ß
Cuser_dnn_layer/user_dnn_layer/user_dnn_2/dense/kernel/part_0/AssignAssign<user_dnn_layer/user_dnn_layer/user_dnn_2/dense/kernel/part_0Wuser_dnn_layer/user_dnn_layer/user_dnn_2/dense/kernel/part_0/Initializer/random_uniform*
_output_shapes
:	@*
T0*O
_classE
CAloc:@user_dnn_layer/user_dnn_layer/user_dnn_2/dense/kernel/part_0

Auser_dnn_layer/user_dnn_layer/user_dnn_2/dense/kernel/part_0/readIdentity<user_dnn_layer/user_dnn_layer/user_dnn_2/dense/kernel/part_0*
_output_shapes
:	@*O
_classE
CAloc:@user_dnn_layer/user_dnn_layer/user_dnn_2/dense/kernel/part_0*
T0
č
Luser_dnn_layer/user_dnn_layer/user_dnn_2/dense/bias/part_0/Initializer/zerosConst*
_output_shapes
:@*M
_classC
A?loc:@user_dnn_layer/user_dnn_layer/user_dnn_2/dense/bias/part_0*
valueB@*    *
dtype0
Ń
:user_dnn_layer/user_dnn_layer/user_dnn_2/dense/bias/part_0
VariableV2*
_output_shapes
:@*M
_classC
A?loc:@user_dnn_layer/user_dnn_layer/user_dnn_2/dense/bias/part_0*
shape:@*
dtype0
É
Auser_dnn_layer/user_dnn_layer/user_dnn_2/dense/bias/part_0/AssignAssign:user_dnn_layer/user_dnn_layer/user_dnn_2/dense/bias/part_0Luser_dnn_layer/user_dnn_layer/user_dnn_2/dense/bias/part_0/Initializer/zeros*
_output_shapes
:@*
T0*M
_classC
A?loc:@user_dnn_layer/user_dnn_layer/user_dnn_2/dense/bias/part_0
ű
?user_dnn_layer/user_dnn_layer/user_dnn_2/dense/bias/part_0/readIdentity:user_dnn_layer/user_dnn_layer/user_dnn_2/dense/bias/part_0*
_output_shapes
:@*
T0*M
_classC
A?loc:@user_dnn_layer/user_dnn_layer/user_dnn_2/dense/bias/part_0
Ž
5user_dnn_layer/user_dnn_layer/user_dnn_2/dense/kernelIdentityAuser_dnn_layer/user_dnn_layer/user_dnn_2/dense/kernel/part_0/read*
T0*
_output_shapes
:	@
×
5user_dnn_layer/user_dnn_layer/user_dnn_2/dense/MatMulMatMul-user_dnn_layer/user_dnn_layer/user_dnn_1/Relu5user_dnn_layer/user_dnn_layer/user_dnn_2/dense/kernel*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@*
T0
Ľ
3user_dnn_layer/user_dnn_layer/user_dnn_2/dense/biasIdentity?user_dnn_layer/user_dnn_layer/user_dnn_2/dense/bias/part_0/read*
T0*
_output_shapes
:@
ß
6user_dnn_layer/user_dnn_layer/user_dnn_2/dense/BiasAddBiasAdd5user_dnn_layer/user_dnn_layer/user_dnn_2/dense/MatMul3user_dnn_layer/user_dnn_layer/user_dnn_2/dense/bias*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@*
T0

Zuser_dnn_layer/user_dnn_layer/user_dnn_2/batch_normalization/gamma/part_0/Initializer/onesConst*\
_classR
PNloc:@user_dnn_layer/user_dnn_layer/user_dnn_2/batch_normalization/gamma/part_0*
valueB@*  ?*
dtype0*
_output_shapes
:@
ď
Iuser_dnn_layer/user_dnn_layer/user_dnn_2/batch_normalization/gamma/part_0
VariableV2*
shape:@*
dtype0*\
_classR
PNloc:@user_dnn_layer/user_dnn_layer/user_dnn_2/batch_normalization/gamma/part_0*
_output_shapes
:@

Puser_dnn_layer/user_dnn_layer/user_dnn_2/batch_normalization/gamma/part_0/AssignAssignIuser_dnn_layer/user_dnn_layer/user_dnn_2/batch_normalization/gamma/part_0Zuser_dnn_layer/user_dnn_layer/user_dnn_2/batch_normalization/gamma/part_0/Initializer/ones*
_output_shapes
:@*
T0*\
_classR
PNloc:@user_dnn_layer/user_dnn_layer/user_dnn_2/batch_normalization/gamma/part_0
¨
Nuser_dnn_layer/user_dnn_layer/user_dnn_2/batch_normalization/gamma/part_0/readIdentityIuser_dnn_layer/user_dnn_layer/user_dnn_2/batch_normalization/gamma/part_0*\
_classR
PNloc:@user_dnn_layer/user_dnn_layer/user_dnn_2/batch_normalization/gamma/part_0*
_output_shapes
:@*
T0

Zuser_dnn_layer/user_dnn_layer/user_dnn_2/batch_normalization/beta/part_0/Initializer/zerosConst*
dtype0*[
_classQ
OMloc:@user_dnn_layer/user_dnn_layer/user_dnn_2/batch_normalization/beta/part_0*
valueB@*    *
_output_shapes
:@
í
Huser_dnn_layer/user_dnn_layer/user_dnn_2/batch_normalization/beta/part_0
VariableV2*
_output_shapes
:@*
dtype0*[
_classQ
OMloc:@user_dnn_layer/user_dnn_layer/user_dnn_2/batch_normalization/beta/part_0*
shape:@

Ouser_dnn_layer/user_dnn_layer/user_dnn_2/batch_normalization/beta/part_0/AssignAssignHuser_dnn_layer/user_dnn_layer/user_dnn_2/batch_normalization/beta/part_0Zuser_dnn_layer/user_dnn_layer/user_dnn_2/batch_normalization/beta/part_0/Initializer/zeros*
_output_shapes
:@*[
_classQ
OMloc:@user_dnn_layer/user_dnn_layer/user_dnn_2/batch_normalization/beta/part_0*
T0
Ľ
Muser_dnn_layer/user_dnn_layer/user_dnn_2/batch_normalization/beta/part_0/readIdentityHuser_dnn_layer/user_dnn_layer/user_dnn_2/batch_normalization/beta/part_0*[
_classQ
OMloc:@user_dnn_layer/user_dnn_layer/user_dnn_2/batch_normalization/beta/part_0*
_output_shapes
:@*
T0

Zuser_dnn_layer/user_dnn_layer/user_dnn_2/batch_normalization/moving_mean/Initializer/zerosConst*
dtype0*
_output_shapes
:@*
valueB@*    *[
_classQ
OMloc:@user_dnn_layer/user_dnn_layer/user_dnn_2/batch_normalization/moving_mean
í
Huser_dnn_layer/user_dnn_layer/user_dnn_2/batch_normalization/moving_mean
VariableV2*
_output_shapes
:@*
dtype0*[
_classQ
OMloc:@user_dnn_layer/user_dnn_layer/user_dnn_2/batch_normalization/moving_mean*
shape:@

Ouser_dnn_layer/user_dnn_layer/user_dnn_2/batch_normalization/moving_mean/AssignAssignHuser_dnn_layer/user_dnn_layer/user_dnn_2/batch_normalization/moving_meanZuser_dnn_layer/user_dnn_layer/user_dnn_2/batch_normalization/moving_mean/Initializer/zeros*
_output_shapes
:@*[
_classQ
OMloc:@user_dnn_layer/user_dnn_layer/user_dnn_2/batch_normalization/moving_mean*
T0
Ľ
Muser_dnn_layer/user_dnn_layer/user_dnn_2/batch_normalization/moving_mean/readIdentityHuser_dnn_layer/user_dnn_layer/user_dnn_2/batch_normalization/moving_mean*[
_classQ
OMloc:@user_dnn_layer/user_dnn_layer/user_dnn_2/batch_normalization/moving_mean*
T0*
_output_shapes
:@

]user_dnn_layer/user_dnn_layer/user_dnn_2/batch_normalization/moving_variance/Initializer/onesConst*
_output_shapes
:@*
valueB@*  ?*
dtype0*_
_classU
SQloc:@user_dnn_layer/user_dnn_layer/user_dnn_2/batch_normalization/moving_variance
ő
Luser_dnn_layer/user_dnn_layer/user_dnn_2/batch_normalization/moving_variance
VariableV2*
shape:@*
dtype0*
_output_shapes
:@*_
_classU
SQloc:@user_dnn_layer/user_dnn_layer/user_dnn_2/batch_normalization/moving_variance

Suser_dnn_layer/user_dnn_layer/user_dnn_2/batch_normalization/moving_variance/AssignAssignLuser_dnn_layer/user_dnn_layer/user_dnn_2/batch_normalization/moving_variance]user_dnn_layer/user_dnn_layer/user_dnn_2/batch_normalization/moving_variance/Initializer/ones*
_output_shapes
:@*_
_classU
SQloc:@user_dnn_layer/user_dnn_layer/user_dnn_2/batch_normalization/moving_variance*
T0
ą
Quser_dnn_layer/user_dnn_layer/user_dnn_2/batch_normalization/moving_variance/readIdentityLuser_dnn_layer/user_dnn_layer/user_dnn_2/batch_normalization/moving_variance*_
_classU
SQloc:@user_dnn_layer/user_dnn_layer/user_dnn_2/batch_normalization/moving_variance*
T0*
_output_shapes
:@
Ľ
[user_dnn_layer/user_dnn_layer/user_dnn_2/batch_normalization/moments/mean/reduction_indicesConst*
valueB: *
dtype0*
_output_shapes
:
 
Iuser_dnn_layer/user_dnn_layer/user_dnn_2/batch_normalization/moments/meanMean6user_dnn_layer/user_dnn_layer/user_dnn_2/dense/BiasAdd[user_dnn_layer/user_dnn_layer/user_dnn_2/batch_normalization/moments/mean/reduction_indices*
T0*
	keep_dims(*
_output_shapes

:@
Ő
Quser_dnn_layer/user_dnn_layer/user_dnn_2/batch_normalization/moments/StopGradientStopGradientIuser_dnn_layer/user_dnn_layer/user_dnn_2/batch_normalization/moments/mean*
_output_shapes

:@*
T0
¨
Vuser_dnn_layer/user_dnn_layer/user_dnn_2/batch_normalization/moments/SquaredDifferenceSquaredDifference6user_dnn_layer/user_dnn_layer/user_dnn_2/dense/BiasAddQuser_dnn_layer/user_dnn_layer/user_dnn_2/batch_normalization/moments/StopGradient*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@*
T0
Š
_user_dnn_layer/user_dnn_layer/user_dnn_2/batch_normalization/moments/variance/reduction_indicesConst*
dtype0*
_output_shapes
:*
valueB: 
Č
Muser_dnn_layer/user_dnn_layer/user_dnn_2/batch_normalization/moments/varianceMeanVuser_dnn_layer/user_dnn_layer/user_dnn_2/batch_normalization/moments/SquaredDifference_user_dnn_layer/user_dnn_layer/user_dnn_2/batch_normalization/moments/variance/reduction_indices*
	keep_dims(*
T0*
_output_shapes

:@
Ţ
Luser_dnn_layer/user_dnn_layer/user_dnn_2/batch_normalization/moments/SqueezeSqueezeIuser_dnn_layer/user_dnn_layer/user_dnn_2/batch_normalization/moments/mean*
_output_shapes
:@*
squeeze_dims
 *
T0
ä
Nuser_dnn_layer/user_dnn_layer/user_dnn_2/batch_normalization/moments/Squeeze_1SqueezeMuser_dnn_layer/user_dnn_layer/user_dnn_2/batch_normalization/moments/variance*
T0*
squeeze_dims
 *
_output_shapes
:@
ô
Ruser_dnn_layer/user_dnn_layer/user_dnn_2/batch_normalization/AssignMovingAvg/decayConst*
_output_shapes
: *[
_classQ
OMloc:@user_dnn_layer/user_dnn_layer/user_dnn_2/batch_normalization/moving_mean*
valueB
 *
×#<*
dtype0
ö
Puser_dnn_layer/user_dnn_layer/user_dnn_2/batch_normalization/AssignMovingAvg/subSubMuser_dnn_layer/user_dnn_layer/user_dnn_2/batch_normalization/moving_mean/readLuser_dnn_layer/user_dnn_layer/user_dnn_2/batch_normalization/moments/Squeeze*
_output_shapes
:@*[
_classQ
OMloc:@user_dnn_layer/user_dnn_layer/user_dnn_2/batch_normalization/moving_mean*
T0
˙
Puser_dnn_layer/user_dnn_layer/user_dnn_2/batch_normalization/AssignMovingAvg/mulMulPuser_dnn_layer/user_dnn_layer/user_dnn_2/batch_normalization/AssignMovingAvg/subRuser_dnn_layer/user_dnn_layer/user_dnn_2/batch_normalization/AssignMovingAvg/decay*
_output_shapes
:@*
T0*[
_classQ
OMloc:@user_dnn_layer/user_dnn_layer/user_dnn_2/batch_normalization/moving_mean
÷
Luser_dnn_layer/user_dnn_layer/user_dnn_2/batch_normalization/AssignMovingAvg	AssignSubHuser_dnn_layer/user_dnn_layer/user_dnn_2/batch_normalization/moving_meanPuser_dnn_layer/user_dnn_layer/user_dnn_2/batch_normalization/AssignMovingAvg/mul*[
_classQ
OMloc:@user_dnn_layer/user_dnn_layer/user_dnn_2/batch_normalization/moving_mean*
T0*
_output_shapes
:@
ú
Tuser_dnn_layer/user_dnn_layer/user_dnn_2/batch_normalization/AssignMovingAvg_1/decayConst*_
_classU
SQloc:@user_dnn_layer/user_dnn_layer/user_dnn_2/batch_normalization/moving_variance*
dtype0*
_output_shapes
: *
valueB
 *
×#<

Ruser_dnn_layer/user_dnn_layer/user_dnn_2/batch_normalization/AssignMovingAvg_1/subSubQuser_dnn_layer/user_dnn_layer/user_dnn_2/batch_normalization/moving_variance/readNuser_dnn_layer/user_dnn_layer/user_dnn_2/batch_normalization/moments/Squeeze_1*
T0*
_output_shapes
:@*_
_classU
SQloc:@user_dnn_layer/user_dnn_layer/user_dnn_2/batch_normalization/moving_variance

Ruser_dnn_layer/user_dnn_layer/user_dnn_2/batch_normalization/AssignMovingAvg_1/mulMulRuser_dnn_layer/user_dnn_layer/user_dnn_2/batch_normalization/AssignMovingAvg_1/subTuser_dnn_layer/user_dnn_layer/user_dnn_2/batch_normalization/AssignMovingAvg_1/decay*
_output_shapes
:@*_
_classU
SQloc:@user_dnn_layer/user_dnn_layer/user_dnn_2/batch_normalization/moving_variance*
T0

Nuser_dnn_layer/user_dnn_layer/user_dnn_2/batch_normalization/AssignMovingAvg_1	AssignSubLuser_dnn_layer/user_dnn_layer/user_dnn_2/batch_normalization/moving_varianceRuser_dnn_layer/user_dnn_layer/user_dnn_2/batch_normalization/AssignMovingAvg_1/mul*
T0*
_output_shapes
:@*_
_classU
SQloc:@user_dnn_layer/user_dnn_layer/user_dnn_2/batch_normalization/moving_variance
Á
Auser_dnn_layer/user_dnn_layer/user_dnn_2/batch_normalization/betaIdentityMuser_dnn_layer/user_dnn_layer/user_dnn_2/batch_normalization/beta/part_0/read*
T0*
_output_shapes
:@
Ă
Buser_dnn_layer/user_dnn_layer/user_dnn_2/batch_normalization/gammaIdentityNuser_dnn_layer/user_dnn_layer/user_dnn_2/batch_normalization/gamma/part_0/read*
T0*
_output_shapes
:@

Luser_dnn_layer/user_dnn_layer/user_dnn_2/batch_normalization/batchnorm/add/yConst*
dtype0*
_output_shapes
: *
valueB
 *o:

Juser_dnn_layer/user_dnn_layer/user_dnn_2/batch_normalization/batchnorm/addAddV2Nuser_dnn_layer/user_dnn_layer/user_dnn_2/batch_normalization/moments/Squeeze_1Luser_dnn_layer/user_dnn_layer/user_dnn_2/batch_normalization/batchnorm/add/y*
_output_shapes
:@*
T0
Ć
Luser_dnn_layer/user_dnn_layer/user_dnn_2/batch_normalization/batchnorm/RsqrtRsqrtJuser_dnn_layer/user_dnn_layer/user_dnn_2/batch_normalization/batchnorm/add*
_output_shapes
:@*
T0

Juser_dnn_layer/user_dnn_layer/user_dnn_2/batch_normalization/batchnorm/mulMulLuser_dnn_layer/user_dnn_layer/user_dnn_2/batch_normalization/batchnorm/RsqrtBuser_dnn_layer/user_dnn_layer/user_dnn_2/batch_normalization/gamma*
T0*
_output_shapes
:@

Luser_dnn_layer/user_dnn_layer/user_dnn_2/batch_normalization/batchnorm/mul_1Mul6user_dnn_layer/user_dnn_layer/user_dnn_2/dense/BiasAddJuser_dnn_layer/user_dnn_layer/user_dnn_2/batch_normalization/batchnorm/mul*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@

Luser_dnn_layer/user_dnn_layer/user_dnn_2/batch_normalization/batchnorm/mul_2MulLuser_dnn_layer/user_dnn_layer/user_dnn_2/batch_normalization/moments/SqueezeJuser_dnn_layer/user_dnn_layer/user_dnn_2/batch_normalization/batchnorm/mul*
T0*
_output_shapes
:@

Juser_dnn_layer/user_dnn_layer/user_dnn_2/batch_normalization/batchnorm/subSubAuser_dnn_layer/user_dnn_layer/user_dnn_2/batch_normalization/betaLuser_dnn_layer/user_dnn_layer/user_dnn_2/batch_normalization/batchnorm/mul_2*
T0*
_output_shapes
:@
Ą
Luser_dnn_layer/user_dnn_layer/user_dnn_2/batch_normalization/batchnorm/add_1AddV2Luser_dnn_layer/user_dnn_layer/user_dnn_2/batch_normalization/batchnorm/mul_1Juser_dnn_layer/user_dnn_layer/user_dnn_2/batch_normalization/batchnorm/sub*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@
ľ
-user_dnn_layer/user_dnn_layer/user_dnn_2/ReluReluLuser_dnn_layer/user_dnn_layer/user_dnn_2/batch_normalization/batchnorm/add_1*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@*
T0
Ł
;user_dnn_layer/user_dnn_layer/user_dnn_2/zero_fraction/SizeSize-user_dnn_layer/user_dnn_layer/user_dnn_2/Relu*
out_type0	*
T0*
_output_shapes
: 

Buser_dnn_layer/user_dnn_layer/user_dnn_2/zero_fraction/LessEqual/yConst*
dtype0	*
valueB	 R˙˙˙˙*
_output_shapes
: 
ď
@user_dnn_layer/user_dnn_layer/user_dnn_2/zero_fraction/LessEqual	LessEqual;user_dnn_layer/user_dnn_layer/user_dnn_2/zero_fraction/SizeBuser_dnn_layer/user_dnn_layer/user_dnn_2/zero_fraction/LessEqual/y*
_output_shapes
: *
T0	
ó
Buser_dnn_layer/user_dnn_layer/user_dnn_2/zero_fraction/cond/SwitchSwitch@user_dnn_layer/user_dnn_layer/user_dnn_2/zero_fraction/LessEqual@user_dnn_layer/user_dnn_layer/user_dnn_2/zero_fraction/LessEqual*
_output_shapes
: : *
T0

ˇ
Duser_dnn_layer/user_dnn_layer/user_dnn_2/zero_fraction/cond/switch_tIdentityDuser_dnn_layer/user_dnn_layer/user_dnn_2/zero_fraction/cond/Switch:1*
_output_shapes
: *
T0

ľ
Duser_dnn_layer/user_dnn_layer/user_dnn_2/zero_fraction/cond/switch_fIdentityBuser_dnn_layer/user_dnn_layer/user_dnn_2/zero_fraction/cond/Switch*
_output_shapes
: *
T0

˛
Cuser_dnn_layer/user_dnn_layer/user_dnn_2/zero_fraction/cond/pred_idIdentity@user_dnn_layer/user_dnn_layer/user_dnn_2/zero_fraction/LessEqual*
_output_shapes
: *
T0

Ű
Ouser_dnn_layer/user_dnn_layer/user_dnn_2/zero_fraction/cond/count_nonzero/zerosConstE^user_dnn_layer/user_dnn_layer/user_dnn_2/zero_fraction/cond/switch_t*
valueB
 *    *
_output_shapes
: *
dtype0
ž
Ruser_dnn_layer/user_dnn_layer/user_dnn_2/zero_fraction/cond/count_nonzero/NotEqualNotEqual[user_dnn_layer/user_dnn_layer/user_dnn_2/zero_fraction/cond/count_nonzero/NotEqual/Switch:1Ouser_dnn_layer/user_dnn_layer/user_dnn_2/zero_fraction/cond/count_nonzero/zeros*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@*
T0
Ţ
Yuser_dnn_layer/user_dnn_layer/user_dnn_2/zero_fraction/cond/count_nonzero/NotEqual/SwitchSwitch-user_dnn_layer/user_dnn_layer/user_dnn_2/ReluCuser_dnn_layer/user_dnn_layer/user_dnn_2/zero_fraction/cond/pred_id*:
_output_shapes(
&:˙˙˙˙˙˙˙˙˙@:˙˙˙˙˙˙˙˙˙@*
T0*@
_class6
42loc:@user_dnn_layer/user_dnn_layer/user_dnn_2/Relu
ë
Nuser_dnn_layer/user_dnn_layer/user_dnn_2/zero_fraction/cond/count_nonzero/CastCastRuser_dnn_layer/user_dnn_layer/user_dnn_2/zero_fraction/cond/count_nonzero/NotEqual*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@*

SrcT0
*

DstT0
ç
Ouser_dnn_layer/user_dnn_layer/user_dnn_2/zero_fraction/cond/count_nonzero/ConstConstE^user_dnn_layer/user_dnn_layer/user_dnn_2/zero_fraction/cond/switch_t*
dtype0*
valueB"       *
_output_shapes
:
 
Wuser_dnn_layer/user_dnn_layer/user_dnn_2/zero_fraction/cond/count_nonzero/nonzero_countSumNuser_dnn_layer/user_dnn_layer/user_dnn_2/zero_fraction/cond/count_nonzero/CastOuser_dnn_layer/user_dnn_layer/user_dnn_2/zero_fraction/cond/count_nonzero/Const*
T0*
_output_shapes
: 
Ń
@user_dnn_layer/user_dnn_layer/user_dnn_2/zero_fraction/cond/CastCastWuser_dnn_layer/user_dnn_layer/user_dnn_2/zero_fraction/cond/count_nonzero/nonzero_count*

SrcT0*
_output_shapes
: *

DstT0	
Ý
Quser_dnn_layer/user_dnn_layer/user_dnn_2/zero_fraction/cond/count_nonzero_1/zerosConstE^user_dnn_layer/user_dnn_layer/user_dnn_2/zero_fraction/cond/switch_f*
_output_shapes
: *
dtype0*
valueB
 *    
Â
Tuser_dnn_layer/user_dnn_layer/user_dnn_2/zero_fraction/cond/count_nonzero_1/NotEqualNotEqual[user_dnn_layer/user_dnn_layer/user_dnn_2/zero_fraction/cond/count_nonzero_1/NotEqual/SwitchQuser_dnn_layer/user_dnn_layer/user_dnn_2/zero_fraction/cond/count_nonzero_1/zeros*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@*
T0
ŕ
[user_dnn_layer/user_dnn_layer/user_dnn_2/zero_fraction/cond/count_nonzero_1/NotEqual/SwitchSwitch-user_dnn_layer/user_dnn_layer/user_dnn_2/ReluCuser_dnn_layer/user_dnn_layer/user_dnn_2/zero_fraction/cond/pred_id*:
_output_shapes(
&:˙˙˙˙˙˙˙˙˙@:˙˙˙˙˙˙˙˙˙@*@
_class6
42loc:@user_dnn_layer/user_dnn_layer/user_dnn_2/Relu*
T0
ď
Puser_dnn_layer/user_dnn_layer/user_dnn_2/zero_fraction/cond/count_nonzero_1/CastCastTuser_dnn_layer/user_dnn_layer/user_dnn_2/zero_fraction/cond/count_nonzero_1/NotEqual*

DstT0	*

SrcT0
*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@
é
Quser_dnn_layer/user_dnn_layer/user_dnn_2/zero_fraction/cond/count_nonzero_1/ConstConstE^user_dnn_layer/user_dnn_layer/user_dnn_2/zero_fraction/cond/switch_f*
valueB"       *
dtype0*
_output_shapes
:
Ś
Yuser_dnn_layer/user_dnn_layer/user_dnn_2/zero_fraction/cond/count_nonzero_1/nonzero_countSumPuser_dnn_layer/user_dnn_layer/user_dnn_2/zero_fraction/cond/count_nonzero_1/CastQuser_dnn_layer/user_dnn_layer/user_dnn_2/zero_fraction/cond/count_nonzero_1/Const*
T0	*
_output_shapes
: 

Auser_dnn_layer/user_dnn_layer/user_dnn_2/zero_fraction/cond/MergeMergeYuser_dnn_layer/user_dnn_layer/user_dnn_2/zero_fraction/cond/count_nonzero_1/nonzero_count@user_dnn_layer/user_dnn_layer/user_dnn_2/zero_fraction/cond/Cast*
N*
T0	*
_output_shapes
: : 
ő
Muser_dnn_layer/user_dnn_layer/user_dnn_2/zero_fraction/counts_to_fraction/subSub;user_dnn_layer/user_dnn_layer/user_dnn_2/zero_fraction/SizeAuser_dnn_layer/user_dnn_layer/user_dnn_2/zero_fraction/cond/Merge*
_output_shapes
: *
T0	
Ő
Nuser_dnn_layer/user_dnn_layer/user_dnn_2/zero_fraction/counts_to_fraction/CastCastMuser_dnn_layer/user_dnn_layer/user_dnn_2/zero_fraction/counts_to_fraction/sub*
_output_shapes
: *

DstT0*

SrcT0	
Ĺ
Puser_dnn_layer/user_dnn_layer/user_dnn_2/zero_fraction/counts_to_fraction/Cast_1Cast;user_dnn_layer/user_dnn_layer/user_dnn_2/zero_fraction/Size*

DstT0*

SrcT0	*
_output_shapes
: 

Quser_dnn_layer/user_dnn_layer/user_dnn_2/zero_fraction/counts_to_fraction/truedivRealDivNuser_dnn_layer/user_dnn_layer/user_dnn_2/zero_fraction/counts_to_fraction/CastPuser_dnn_layer/user_dnn_layer/user_dnn_2/zero_fraction/counts_to_fraction/Cast_1*
T0*
_output_shapes
: 
ż
?user_dnn_layer/user_dnn_layer/user_dnn_2/zero_fraction/fractionIdentityQuser_dnn_layer/user_dnn_layer/user_dnn_2/zero_fraction/counts_to_fraction/truediv*
_output_shapes
: *
T0

nuser_dnn_layer/user_dnn_layer/user_dnn_2/user_dnn_layer/user_dnn_layer/user_dnn_2/fraction_of_zero_values/tagsConst*
_output_shapes
: *z
valueqBo Biuser_dnn_layer/user_dnn_layer/user_dnn_2/user_dnn_layer/user_dnn_layer/user_dnn_2/fraction_of_zero_values*
dtype0
Ě
iuser_dnn_layer/user_dnn_layer/user_dnn_2/user_dnn_layer/user_dnn_layer/user_dnn_2/fraction_of_zero_valuesScalarSummarynuser_dnn_layer/user_dnn_layer/user_dnn_2/user_dnn_layer/user_dnn_layer/user_dnn_2/fraction_of_zero_values/tags?user_dnn_layer/user_dnn_layer/user_dnn_2/zero_fraction/fraction*
T0*
_output_shapes
: 
ý
`user_dnn_layer/user_dnn_layer/user_dnn_2/user_dnn_layer/user_dnn_layer/user_dnn_2/activation/tagConst*m
valuedBb B\user_dnn_layer/user_dnn_layer/user_dnn_2/user_dnn_layer/user_dnn_layer/user_dnn_2/activation*
dtype0*
_output_shapes
: 

\user_dnn_layer/user_dnn_layer/user_dnn_2/user_dnn_layer/user_dnn_layer/user_dnn_2/activationHistogramSummary`user_dnn_layer/user_dnn_layer/user_dnn_2/user_dnn_layer/user_dnn_layer/user_dnn_2/activation/tag-user_dnn_layer/user_dnn_layer/user_dnn_2/Relu*
_output_shapes
: 
˙
]user_dnn_layer/user_dnn_layer/user_dnn_3/dense/kernel/part_0/Initializer/random_uniform/shapeConst*O
_classE
CAloc:@user_dnn_layer/user_dnn_layer/user_dnn_3/dense/kernel/part_0*
valueB"@       *
_output_shapes
:*
dtype0
ń
[user_dnn_layer/user_dnn_layer/user_dnn_3/dense/kernel/part_0/Initializer/random_uniform/minConst*O
_classE
CAloc:@user_dnn_layer/user_dnn_layer/user_dnn_3/dense/kernel/part_0*
dtype0*
valueB
 *  ž*
_output_shapes
: 
ń
[user_dnn_layer/user_dnn_layer/user_dnn_3/dense/kernel/part_0/Initializer/random_uniform/maxConst*O
_classE
CAloc:@user_dnn_layer/user_dnn_layer/user_dnn_3/dense/kernel/part_0*
valueB
 *  >*
_output_shapes
: *
dtype0
Ü
euser_dnn_layer/user_dnn_layer/user_dnn_3/dense/kernel/part_0/Initializer/random_uniform/RandomUniformRandomUniform]user_dnn_layer/user_dnn_layer/user_dnn_3/dense/kernel/part_0/Initializer/random_uniform/shape*
T0*
_output_shapes

:@ *
dtype0*O
_classE
CAloc:@user_dnn_layer/user_dnn_layer/user_dnn_3/dense/kernel/part_0

[user_dnn_layer/user_dnn_layer/user_dnn_3/dense/kernel/part_0/Initializer/random_uniform/subSub[user_dnn_layer/user_dnn_layer/user_dnn_3/dense/kernel/part_0/Initializer/random_uniform/max[user_dnn_layer/user_dnn_layer/user_dnn_3/dense/kernel/part_0/Initializer/random_uniform/min*
T0*O
_classE
CAloc:@user_dnn_layer/user_dnn_layer/user_dnn_3/dense/kernel/part_0*
_output_shapes
: 
 
[user_dnn_layer/user_dnn_layer/user_dnn_3/dense/kernel/part_0/Initializer/random_uniform/mulMuleuser_dnn_layer/user_dnn_layer/user_dnn_3/dense/kernel/part_0/Initializer/random_uniform/RandomUniform[user_dnn_layer/user_dnn_layer/user_dnn_3/dense/kernel/part_0/Initializer/random_uniform/sub*O
_classE
CAloc:@user_dnn_layer/user_dnn_layer/user_dnn_3/dense/kernel/part_0*
T0*
_output_shapes

:@ 

Wuser_dnn_layer/user_dnn_layer/user_dnn_3/dense/kernel/part_0/Initializer/random_uniformAdd[user_dnn_layer/user_dnn_layer/user_dnn_3/dense/kernel/part_0/Initializer/random_uniform/mul[user_dnn_layer/user_dnn_layer/user_dnn_3/dense/kernel/part_0/Initializer/random_uniform/min*
T0*O
_classE
CAloc:@user_dnn_layer/user_dnn_layer/user_dnn_3/dense/kernel/part_0*
_output_shapes

:@ 
Ý
<user_dnn_layer/user_dnn_layer/user_dnn_3/dense/kernel/part_0
VariableV2*O
_classE
CAloc:@user_dnn_layer/user_dnn_layer/user_dnn_3/dense/kernel/part_0*
dtype0*
shape
:@ *
_output_shapes

:@ 
Ţ
Cuser_dnn_layer/user_dnn_layer/user_dnn_3/dense/kernel/part_0/AssignAssign<user_dnn_layer/user_dnn_layer/user_dnn_3/dense/kernel/part_0Wuser_dnn_layer/user_dnn_layer/user_dnn_3/dense/kernel/part_0/Initializer/random_uniform*
T0*
_output_shapes

:@ *O
_classE
CAloc:@user_dnn_layer/user_dnn_layer/user_dnn_3/dense/kernel/part_0

Auser_dnn_layer/user_dnn_layer/user_dnn_3/dense/kernel/part_0/readIdentity<user_dnn_layer/user_dnn_layer/user_dnn_3/dense/kernel/part_0*O
_classE
CAloc:@user_dnn_layer/user_dnn_layer/user_dnn_3/dense/kernel/part_0*
_output_shapes

:@ *
T0
č
Luser_dnn_layer/user_dnn_layer/user_dnn_3/dense/bias/part_0/Initializer/zerosConst*
dtype0*M
_classC
A?loc:@user_dnn_layer/user_dnn_layer/user_dnn_3/dense/bias/part_0*
valueB *    *
_output_shapes
: 
Ń
:user_dnn_layer/user_dnn_layer/user_dnn_3/dense/bias/part_0
VariableV2*
_output_shapes
: *
dtype0*
shape: *M
_classC
A?loc:@user_dnn_layer/user_dnn_layer/user_dnn_3/dense/bias/part_0
É
Auser_dnn_layer/user_dnn_layer/user_dnn_3/dense/bias/part_0/AssignAssign:user_dnn_layer/user_dnn_layer/user_dnn_3/dense/bias/part_0Luser_dnn_layer/user_dnn_layer/user_dnn_3/dense/bias/part_0/Initializer/zeros*M
_classC
A?loc:@user_dnn_layer/user_dnn_layer/user_dnn_3/dense/bias/part_0*
T0*
_output_shapes
: 
ű
?user_dnn_layer/user_dnn_layer/user_dnn_3/dense/bias/part_0/readIdentity:user_dnn_layer/user_dnn_layer/user_dnn_3/dense/bias/part_0*
_output_shapes
: *
T0*M
_classC
A?loc:@user_dnn_layer/user_dnn_layer/user_dnn_3/dense/bias/part_0
­
5user_dnn_layer/user_dnn_layer/user_dnn_3/dense/kernelIdentityAuser_dnn_layer/user_dnn_layer/user_dnn_3/dense/kernel/part_0/read*
T0*
_output_shapes

:@ 
×
5user_dnn_layer/user_dnn_layer/user_dnn_3/dense/MatMulMatMul-user_dnn_layer/user_dnn_layer/user_dnn_2/Relu5user_dnn_layer/user_dnn_layer/user_dnn_3/dense/kernel*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
Ľ
3user_dnn_layer/user_dnn_layer/user_dnn_3/dense/biasIdentity?user_dnn_layer/user_dnn_layer/user_dnn_3/dense/bias/part_0/read*
T0*
_output_shapes
: 
ß
6user_dnn_layer/user_dnn_layer/user_dnn_3/dense/BiasAddBiasAdd5user_dnn_layer/user_dnn_layer/user_dnn_3/dense/MatMul3user_dnn_layer/user_dnn_layer/user_dnn_3/dense/bias*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
Ź
;user_dnn_layer/user_dnn_layer/user_dnn_3/zero_fraction/SizeSize6user_dnn_layer/user_dnn_layer/user_dnn_3/dense/BiasAdd*
out_type0	*
T0*
_output_shapes
: 

Buser_dnn_layer/user_dnn_layer/user_dnn_3/zero_fraction/LessEqual/yConst*
valueB	 R˙˙˙˙*
dtype0	*
_output_shapes
: 
ď
@user_dnn_layer/user_dnn_layer/user_dnn_3/zero_fraction/LessEqual	LessEqual;user_dnn_layer/user_dnn_layer/user_dnn_3/zero_fraction/SizeBuser_dnn_layer/user_dnn_layer/user_dnn_3/zero_fraction/LessEqual/y*
_output_shapes
: *
T0	
ó
Buser_dnn_layer/user_dnn_layer/user_dnn_3/zero_fraction/cond/SwitchSwitch@user_dnn_layer/user_dnn_layer/user_dnn_3/zero_fraction/LessEqual@user_dnn_layer/user_dnn_layer/user_dnn_3/zero_fraction/LessEqual*
_output_shapes
: : *
T0

ˇ
Duser_dnn_layer/user_dnn_layer/user_dnn_3/zero_fraction/cond/switch_tIdentityDuser_dnn_layer/user_dnn_layer/user_dnn_3/zero_fraction/cond/Switch:1*
_output_shapes
: *
T0

ľ
Duser_dnn_layer/user_dnn_layer/user_dnn_3/zero_fraction/cond/switch_fIdentityBuser_dnn_layer/user_dnn_layer/user_dnn_3/zero_fraction/cond/Switch*
T0
*
_output_shapes
: 
˛
Cuser_dnn_layer/user_dnn_layer/user_dnn_3/zero_fraction/cond/pred_idIdentity@user_dnn_layer/user_dnn_layer/user_dnn_3/zero_fraction/LessEqual*
_output_shapes
: *
T0

Ű
Ouser_dnn_layer/user_dnn_layer/user_dnn_3/zero_fraction/cond/count_nonzero/zerosConstE^user_dnn_layer/user_dnn_layer/user_dnn_3/zero_fraction/cond/switch_t*
dtype0*
_output_shapes
: *
valueB
 *    
ž
Ruser_dnn_layer/user_dnn_layer/user_dnn_3/zero_fraction/cond/count_nonzero/NotEqualNotEqual[user_dnn_layer/user_dnn_layer/user_dnn_3/zero_fraction/cond/count_nonzero/NotEqual/Switch:1Ouser_dnn_layer/user_dnn_layer/user_dnn_3/zero_fraction/cond/count_nonzero/zeros*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
đ
Yuser_dnn_layer/user_dnn_layer/user_dnn_3/zero_fraction/cond/count_nonzero/NotEqual/SwitchSwitch6user_dnn_layer/user_dnn_layer/user_dnn_3/dense/BiasAddCuser_dnn_layer/user_dnn_layer/user_dnn_3/zero_fraction/cond/pred_id*
T0*I
_class?
=;loc:@user_dnn_layer/user_dnn_layer/user_dnn_3/dense/BiasAdd*:
_output_shapes(
&:˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ 
ë
Nuser_dnn_layer/user_dnn_layer/user_dnn_3/zero_fraction/cond/count_nonzero/CastCastRuser_dnn_layer/user_dnn_layer/user_dnn_3/zero_fraction/cond/count_nonzero/NotEqual*

SrcT0
*

DstT0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
ç
Ouser_dnn_layer/user_dnn_layer/user_dnn_3/zero_fraction/cond/count_nonzero/ConstConstE^user_dnn_layer/user_dnn_layer/user_dnn_3/zero_fraction/cond/switch_t*
valueB"       *
dtype0*
_output_shapes
:
 
Wuser_dnn_layer/user_dnn_layer/user_dnn_3/zero_fraction/cond/count_nonzero/nonzero_countSumNuser_dnn_layer/user_dnn_layer/user_dnn_3/zero_fraction/cond/count_nonzero/CastOuser_dnn_layer/user_dnn_layer/user_dnn_3/zero_fraction/cond/count_nonzero/Const*
_output_shapes
: *
T0
Ń
@user_dnn_layer/user_dnn_layer/user_dnn_3/zero_fraction/cond/CastCastWuser_dnn_layer/user_dnn_layer/user_dnn_3/zero_fraction/cond/count_nonzero/nonzero_count*
_output_shapes
: *

SrcT0*

DstT0	
Ý
Quser_dnn_layer/user_dnn_layer/user_dnn_3/zero_fraction/cond/count_nonzero_1/zerosConstE^user_dnn_layer/user_dnn_layer/user_dnn_3/zero_fraction/cond/switch_f*
_output_shapes
: *
valueB
 *    *
dtype0
Â
Tuser_dnn_layer/user_dnn_layer/user_dnn_3/zero_fraction/cond/count_nonzero_1/NotEqualNotEqual[user_dnn_layer/user_dnn_layer/user_dnn_3/zero_fraction/cond/count_nonzero_1/NotEqual/SwitchQuser_dnn_layer/user_dnn_layer/user_dnn_3/zero_fraction/cond/count_nonzero_1/zeros*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
ň
[user_dnn_layer/user_dnn_layer/user_dnn_3/zero_fraction/cond/count_nonzero_1/NotEqual/SwitchSwitch6user_dnn_layer/user_dnn_layer/user_dnn_3/dense/BiasAddCuser_dnn_layer/user_dnn_layer/user_dnn_3/zero_fraction/cond/pred_id*I
_class?
=;loc:@user_dnn_layer/user_dnn_layer/user_dnn_3/dense/BiasAdd*:
_output_shapes(
&:˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ *
T0
ď
Puser_dnn_layer/user_dnn_layer/user_dnn_3/zero_fraction/cond/count_nonzero_1/CastCastTuser_dnn_layer/user_dnn_layer/user_dnn_3/zero_fraction/cond/count_nonzero_1/NotEqual*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ *

DstT0	*

SrcT0

é
Quser_dnn_layer/user_dnn_layer/user_dnn_3/zero_fraction/cond/count_nonzero_1/ConstConstE^user_dnn_layer/user_dnn_layer/user_dnn_3/zero_fraction/cond/switch_f*
dtype0*
_output_shapes
:*
valueB"       
Ś
Yuser_dnn_layer/user_dnn_layer/user_dnn_3/zero_fraction/cond/count_nonzero_1/nonzero_countSumPuser_dnn_layer/user_dnn_layer/user_dnn_3/zero_fraction/cond/count_nonzero_1/CastQuser_dnn_layer/user_dnn_layer/user_dnn_3/zero_fraction/cond/count_nonzero_1/Const*
T0	*
_output_shapes
: 

Auser_dnn_layer/user_dnn_layer/user_dnn_3/zero_fraction/cond/MergeMergeYuser_dnn_layer/user_dnn_layer/user_dnn_3/zero_fraction/cond/count_nonzero_1/nonzero_count@user_dnn_layer/user_dnn_layer/user_dnn_3/zero_fraction/cond/Cast*
N*
T0	*
_output_shapes
: : 
ő
Muser_dnn_layer/user_dnn_layer/user_dnn_3/zero_fraction/counts_to_fraction/subSub;user_dnn_layer/user_dnn_layer/user_dnn_3/zero_fraction/SizeAuser_dnn_layer/user_dnn_layer/user_dnn_3/zero_fraction/cond/Merge*
T0	*
_output_shapes
: 
Ő
Nuser_dnn_layer/user_dnn_layer/user_dnn_3/zero_fraction/counts_to_fraction/CastCastMuser_dnn_layer/user_dnn_layer/user_dnn_3/zero_fraction/counts_to_fraction/sub*

SrcT0	*
_output_shapes
: *

DstT0
Ĺ
Puser_dnn_layer/user_dnn_layer/user_dnn_3/zero_fraction/counts_to_fraction/Cast_1Cast;user_dnn_layer/user_dnn_layer/user_dnn_3/zero_fraction/Size*

DstT0*
_output_shapes
: *

SrcT0	

Quser_dnn_layer/user_dnn_layer/user_dnn_3/zero_fraction/counts_to_fraction/truedivRealDivNuser_dnn_layer/user_dnn_layer/user_dnn_3/zero_fraction/counts_to_fraction/CastPuser_dnn_layer/user_dnn_layer/user_dnn_3/zero_fraction/counts_to_fraction/Cast_1*
T0*
_output_shapes
: 
ż
?user_dnn_layer/user_dnn_layer/user_dnn_3/zero_fraction/fractionIdentityQuser_dnn_layer/user_dnn_layer/user_dnn_3/zero_fraction/counts_to_fraction/truediv*
_output_shapes
: *
T0

nuser_dnn_layer/user_dnn_layer/user_dnn_3/user_dnn_layer/user_dnn_layer/user_dnn_3/fraction_of_zero_values/tagsConst*
_output_shapes
: *z
valueqBo Biuser_dnn_layer/user_dnn_layer/user_dnn_3/user_dnn_layer/user_dnn_layer/user_dnn_3/fraction_of_zero_values*
dtype0
Ě
iuser_dnn_layer/user_dnn_layer/user_dnn_3/user_dnn_layer/user_dnn_layer/user_dnn_3/fraction_of_zero_valuesScalarSummarynuser_dnn_layer/user_dnn_layer/user_dnn_3/user_dnn_layer/user_dnn_layer/user_dnn_3/fraction_of_zero_values/tags?user_dnn_layer/user_dnn_layer/user_dnn_3/zero_fraction/fraction*
_output_shapes
: *
T0
ý
`user_dnn_layer/user_dnn_layer/user_dnn_3/user_dnn_layer/user_dnn_layer/user_dnn_3/activation/tagConst*m
valuedBb B\user_dnn_layer/user_dnn_layer/user_dnn_3/user_dnn_layer/user_dnn_layer/user_dnn_3/activation*
dtype0*
_output_shapes
: 
˘
\user_dnn_layer/user_dnn_layer/user_dnn_3/user_dnn_layer/user_dnn_layer/user_dnn_3/activationHistogramSummary`user_dnn_layer/user_dnn_layer/user_dnn_3/user_dnn_layer/user_dnn_layer/user_dnn_3/activation/tag6user_dnn_layer/user_dnn_layer/user_dnn_3/dense/BiasAdd*
_output_shapes
: 
˙
]item_dnn_layer/item_dnn_layer/item_dnn_0/dense/kernel/part_0/Initializer/random_uniform/shapeConst*
valueB"      *O
_classE
CAloc:@item_dnn_layer/item_dnn_layer/item_dnn_0/dense/kernel/part_0*
_output_shapes
:*
dtype0
ń
[item_dnn_layer/item_dnn_layer/item_dnn_0/dense/kernel/part_0/Initializer/random_uniform/minConst*O
_classE
CAloc:@item_dnn_layer/item_dnn_layer/item_dnn_0/dense/kernel/part_0*
valueB
 *   ž*
_output_shapes
: *
dtype0
ń
[item_dnn_layer/item_dnn_layer/item_dnn_0/dense/kernel/part_0/Initializer/random_uniform/maxConst*O
_classE
CAloc:@item_dnn_layer/item_dnn_layer/item_dnn_0/dense/kernel/part_0*
valueB
 *   >*
dtype0*
_output_shapes
: 
Ţ
eitem_dnn_layer/item_dnn_layer/item_dnn_0/dense/kernel/part_0/Initializer/random_uniform/RandomUniformRandomUniform]item_dnn_layer/item_dnn_layer/item_dnn_0/dense/kernel/part_0/Initializer/random_uniform/shape*
dtype0*
T0*O
_classE
CAloc:@item_dnn_layer/item_dnn_layer/item_dnn_0/dense/kernel/part_0* 
_output_shapes
:


[item_dnn_layer/item_dnn_layer/item_dnn_0/dense/kernel/part_0/Initializer/random_uniform/subSub[item_dnn_layer/item_dnn_layer/item_dnn_0/dense/kernel/part_0/Initializer/random_uniform/max[item_dnn_layer/item_dnn_layer/item_dnn_0/dense/kernel/part_0/Initializer/random_uniform/min*
T0*O
_classE
CAloc:@item_dnn_layer/item_dnn_layer/item_dnn_0/dense/kernel/part_0*
_output_shapes
: 
˘
[item_dnn_layer/item_dnn_layer/item_dnn_0/dense/kernel/part_0/Initializer/random_uniform/mulMuleitem_dnn_layer/item_dnn_layer/item_dnn_0/dense/kernel/part_0/Initializer/random_uniform/RandomUniform[item_dnn_layer/item_dnn_layer/item_dnn_0/dense/kernel/part_0/Initializer/random_uniform/sub*O
_classE
CAloc:@item_dnn_layer/item_dnn_layer/item_dnn_0/dense/kernel/part_0* 
_output_shapes
:
*
T0

Witem_dnn_layer/item_dnn_layer/item_dnn_0/dense/kernel/part_0/Initializer/random_uniformAdd[item_dnn_layer/item_dnn_layer/item_dnn_0/dense/kernel/part_0/Initializer/random_uniform/mul[item_dnn_layer/item_dnn_layer/item_dnn_0/dense/kernel/part_0/Initializer/random_uniform/min*O
_classE
CAloc:@item_dnn_layer/item_dnn_layer/item_dnn_0/dense/kernel/part_0*
T0* 
_output_shapes
:

á
<item_dnn_layer/item_dnn_layer/item_dnn_0/dense/kernel/part_0
VariableV2*
shape:
*
dtype0*O
_classE
CAloc:@item_dnn_layer/item_dnn_layer/item_dnn_0/dense/kernel/part_0* 
_output_shapes
:

ŕ
Citem_dnn_layer/item_dnn_layer/item_dnn_0/dense/kernel/part_0/AssignAssign<item_dnn_layer/item_dnn_layer/item_dnn_0/dense/kernel/part_0Witem_dnn_layer/item_dnn_layer/item_dnn_0/dense/kernel/part_0/Initializer/random_uniform*O
_classE
CAloc:@item_dnn_layer/item_dnn_layer/item_dnn_0/dense/kernel/part_0*
T0* 
_output_shapes
:


Aitem_dnn_layer/item_dnn_layer/item_dnn_0/dense/kernel/part_0/readIdentity<item_dnn_layer/item_dnn_layer/item_dnn_0/dense/kernel/part_0*
T0*O
_classE
CAloc:@item_dnn_layer/item_dnn_layer/item_dnn_0/dense/kernel/part_0* 
_output_shapes
:

ę
Litem_dnn_layer/item_dnn_layer/item_dnn_0/dense/bias/part_0/Initializer/zerosConst*M
_classC
A?loc:@item_dnn_layer/item_dnn_layer/item_dnn_0/dense/bias/part_0*
valueB*    *
dtype0*
_output_shapes	
:
Ó
:item_dnn_layer/item_dnn_layer/item_dnn_0/dense/bias/part_0
VariableV2*
_output_shapes	
:*
shape:*M
_classC
A?loc:@item_dnn_layer/item_dnn_layer/item_dnn_0/dense/bias/part_0*
dtype0
Ę
Aitem_dnn_layer/item_dnn_layer/item_dnn_0/dense/bias/part_0/AssignAssign:item_dnn_layer/item_dnn_layer/item_dnn_0/dense/bias/part_0Litem_dnn_layer/item_dnn_layer/item_dnn_0/dense/bias/part_0/Initializer/zeros*
_output_shapes	
:*
T0*M
_classC
A?loc:@item_dnn_layer/item_dnn_layer/item_dnn_0/dense/bias/part_0
ü
?item_dnn_layer/item_dnn_layer/item_dnn_0/dense/bias/part_0/readIdentity:item_dnn_layer/item_dnn_layer/item_dnn_0/dense/bias/part_0*M
_classC
A?loc:@item_dnn_layer/item_dnn_layer/item_dnn_0/dense/bias/part_0*
T0*
_output_shapes	
:
Ż
5item_dnn_layer/item_dnn_layer/item_dnn_0/dense/kernelIdentityAitem_dnn_layer/item_dnn_layer/item_dnn_0/dense/kernel/part_0/read*
T0* 
_output_shapes
:

Ë
5item_dnn_layer/item_dnn_layer/item_dnn_0/dense/MatMulMatMul input_layer/input_layer_1/concat5item_dnn_layer/item_dnn_layer/item_dnn_0/dense/kernel*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ś
3item_dnn_layer/item_dnn_layer/item_dnn_0/dense/biasIdentity?item_dnn_layer/item_dnn_layer/item_dnn_0/dense/bias/part_0/read*
_output_shapes	
:*
T0
ŕ
6item_dnn_layer/item_dnn_layer/item_dnn_0/dense/BiasAddBiasAdd5item_dnn_layer/item_dnn_layer/item_dnn_0/dense/MatMul3item_dnn_layer/item_dnn_layer/item_dnn_0/dense/bias*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

Zitem_dnn_layer/item_dnn_layer/item_dnn_0/batch_normalization/gamma/part_0/Initializer/onesConst*
_output_shapes	
:*\
_classR
PNloc:@item_dnn_layer/item_dnn_layer/item_dnn_0/batch_normalization/gamma/part_0*
valueB*  ?*
dtype0
ń
Iitem_dnn_layer/item_dnn_layer/item_dnn_0/batch_normalization/gamma/part_0
VariableV2*
shape:*
_output_shapes	
:*
dtype0*\
_classR
PNloc:@item_dnn_layer/item_dnn_layer/item_dnn_0/batch_normalization/gamma/part_0

Pitem_dnn_layer/item_dnn_layer/item_dnn_0/batch_normalization/gamma/part_0/AssignAssignIitem_dnn_layer/item_dnn_layer/item_dnn_0/batch_normalization/gamma/part_0Zitem_dnn_layer/item_dnn_layer/item_dnn_0/batch_normalization/gamma/part_0/Initializer/ones*\
_classR
PNloc:@item_dnn_layer/item_dnn_layer/item_dnn_0/batch_normalization/gamma/part_0*
_output_shapes	
:*
T0
Š
Nitem_dnn_layer/item_dnn_layer/item_dnn_0/batch_normalization/gamma/part_0/readIdentityIitem_dnn_layer/item_dnn_layer/item_dnn_0/batch_normalization/gamma/part_0*
_output_shapes	
:*
T0*\
_classR
PNloc:@item_dnn_layer/item_dnn_layer/item_dnn_0/batch_normalization/gamma/part_0

Zitem_dnn_layer/item_dnn_layer/item_dnn_0/batch_normalization/beta/part_0/Initializer/zerosConst*
valueB*    *
dtype0*
_output_shapes	
:*[
_classQ
OMloc:@item_dnn_layer/item_dnn_layer/item_dnn_0/batch_normalization/beta/part_0
ď
Hitem_dnn_layer/item_dnn_layer/item_dnn_0/batch_normalization/beta/part_0
VariableV2*
dtype0*
shape:*
_output_shapes	
:*[
_classQ
OMloc:@item_dnn_layer/item_dnn_layer/item_dnn_0/batch_normalization/beta/part_0

Oitem_dnn_layer/item_dnn_layer/item_dnn_0/batch_normalization/beta/part_0/AssignAssignHitem_dnn_layer/item_dnn_layer/item_dnn_0/batch_normalization/beta/part_0Zitem_dnn_layer/item_dnn_layer/item_dnn_0/batch_normalization/beta/part_0/Initializer/zeros*
_output_shapes	
:*[
_classQ
OMloc:@item_dnn_layer/item_dnn_layer/item_dnn_0/batch_normalization/beta/part_0*
T0
Ś
Mitem_dnn_layer/item_dnn_layer/item_dnn_0/batch_normalization/beta/part_0/readIdentityHitem_dnn_layer/item_dnn_layer/item_dnn_0/batch_normalization/beta/part_0*
T0*
_output_shapes	
:*[
_classQ
OMloc:@item_dnn_layer/item_dnn_layer/item_dnn_0/batch_normalization/beta/part_0

Zitem_dnn_layer/item_dnn_layer/item_dnn_0/batch_normalization/moving_mean/Initializer/zerosConst*
dtype0*
valueB*    *
_output_shapes	
:*[
_classQ
OMloc:@item_dnn_layer/item_dnn_layer/item_dnn_0/batch_normalization/moving_mean
ď
Hitem_dnn_layer/item_dnn_layer/item_dnn_0/batch_normalization/moving_mean
VariableV2*
shape:*
_output_shapes	
:*
dtype0*[
_classQ
OMloc:@item_dnn_layer/item_dnn_layer/item_dnn_0/batch_normalization/moving_mean

Oitem_dnn_layer/item_dnn_layer/item_dnn_0/batch_normalization/moving_mean/AssignAssignHitem_dnn_layer/item_dnn_layer/item_dnn_0/batch_normalization/moving_meanZitem_dnn_layer/item_dnn_layer/item_dnn_0/batch_normalization/moving_mean/Initializer/zeros*
T0*[
_classQ
OMloc:@item_dnn_layer/item_dnn_layer/item_dnn_0/batch_normalization/moving_mean*
_output_shapes	
:
Ś
Mitem_dnn_layer/item_dnn_layer/item_dnn_0/batch_normalization/moving_mean/readIdentityHitem_dnn_layer/item_dnn_layer/item_dnn_0/batch_normalization/moving_mean*
T0*[
_classQ
OMloc:@item_dnn_layer/item_dnn_layer/item_dnn_0/batch_normalization/moving_mean*
_output_shapes	
:

]item_dnn_layer/item_dnn_layer/item_dnn_0/batch_normalization/moving_variance/Initializer/onesConst*
_output_shapes	
:*
valueB*  ?*
dtype0*_
_classU
SQloc:@item_dnn_layer/item_dnn_layer/item_dnn_0/batch_normalization/moving_variance
÷
Litem_dnn_layer/item_dnn_layer/item_dnn_0/batch_normalization/moving_variance
VariableV2*
dtype0*_
_classU
SQloc:@item_dnn_layer/item_dnn_layer/item_dnn_0/batch_normalization/moving_variance*
shape:*
_output_shapes	
:

Sitem_dnn_layer/item_dnn_layer/item_dnn_0/batch_normalization/moving_variance/AssignAssignLitem_dnn_layer/item_dnn_layer/item_dnn_0/batch_normalization/moving_variance]item_dnn_layer/item_dnn_layer/item_dnn_0/batch_normalization/moving_variance/Initializer/ones*
_output_shapes	
:*_
_classU
SQloc:@item_dnn_layer/item_dnn_layer/item_dnn_0/batch_normalization/moving_variance*
T0
˛
Qitem_dnn_layer/item_dnn_layer/item_dnn_0/batch_normalization/moving_variance/readIdentityLitem_dnn_layer/item_dnn_layer/item_dnn_0/batch_normalization/moving_variance*
_output_shapes	
:*
T0*_
_classU
SQloc:@item_dnn_layer/item_dnn_layer/item_dnn_0/batch_normalization/moving_variance
Ľ
[item_dnn_layer/item_dnn_layer/item_dnn_0/batch_normalization/moments/mean/reduction_indicesConst*
dtype0*
_output_shapes
:*
valueB: 
Ą
Iitem_dnn_layer/item_dnn_layer/item_dnn_0/batch_normalization/moments/meanMean6item_dnn_layer/item_dnn_layer/item_dnn_0/dense/BiasAdd[item_dnn_layer/item_dnn_layer/item_dnn_0/batch_normalization/moments/mean/reduction_indices*
_output_shapes
:	*
	keep_dims(*
T0
Ö
Qitem_dnn_layer/item_dnn_layer/item_dnn_0/batch_normalization/moments/StopGradientStopGradientIitem_dnn_layer/item_dnn_layer/item_dnn_0/batch_normalization/moments/mean*
_output_shapes
:	*
T0
Š
Vitem_dnn_layer/item_dnn_layer/item_dnn_0/batch_normalization/moments/SquaredDifferenceSquaredDifference6item_dnn_layer/item_dnn_layer/item_dnn_0/dense/BiasAddQitem_dnn_layer/item_dnn_layer/item_dnn_0/batch_normalization/moments/StopGradient*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Š
_item_dnn_layer/item_dnn_layer/item_dnn_0/batch_normalization/moments/variance/reduction_indicesConst*
dtype0*
_output_shapes
:*
valueB: 
É
Mitem_dnn_layer/item_dnn_layer/item_dnn_0/batch_normalization/moments/varianceMeanVitem_dnn_layer/item_dnn_layer/item_dnn_0/batch_normalization/moments/SquaredDifference_item_dnn_layer/item_dnn_layer/item_dnn_0/batch_normalization/moments/variance/reduction_indices*
T0*
	keep_dims(*
_output_shapes
:	
ß
Litem_dnn_layer/item_dnn_layer/item_dnn_0/batch_normalization/moments/SqueezeSqueezeIitem_dnn_layer/item_dnn_layer/item_dnn_0/batch_normalization/moments/mean*
squeeze_dims
 *
_output_shapes	
:*
T0
ĺ
Nitem_dnn_layer/item_dnn_layer/item_dnn_0/batch_normalization/moments/Squeeze_1SqueezeMitem_dnn_layer/item_dnn_layer/item_dnn_0/batch_normalization/moments/variance*
T0*
_output_shapes	
:*
squeeze_dims
 
ô
Ritem_dnn_layer/item_dnn_layer/item_dnn_0/batch_normalization/AssignMovingAvg/decayConst*
valueB
 *
×#<*[
_classQ
OMloc:@item_dnn_layer/item_dnn_layer/item_dnn_0/batch_normalization/moving_mean*
_output_shapes
: *
dtype0
÷
Pitem_dnn_layer/item_dnn_layer/item_dnn_0/batch_normalization/AssignMovingAvg/subSubMitem_dnn_layer/item_dnn_layer/item_dnn_0/batch_normalization/moving_mean/readLitem_dnn_layer/item_dnn_layer/item_dnn_0/batch_normalization/moments/Squeeze*
_output_shapes	
:*[
_classQ
OMloc:@item_dnn_layer/item_dnn_layer/item_dnn_0/batch_normalization/moving_mean*
T0

Pitem_dnn_layer/item_dnn_layer/item_dnn_0/batch_normalization/AssignMovingAvg/mulMulPitem_dnn_layer/item_dnn_layer/item_dnn_0/batch_normalization/AssignMovingAvg/subRitem_dnn_layer/item_dnn_layer/item_dnn_0/batch_normalization/AssignMovingAvg/decay*
_output_shapes	
:*
T0*[
_classQ
OMloc:@item_dnn_layer/item_dnn_layer/item_dnn_0/batch_normalization/moving_mean
ř
Litem_dnn_layer/item_dnn_layer/item_dnn_0/batch_normalization/AssignMovingAvg	AssignSubHitem_dnn_layer/item_dnn_layer/item_dnn_0/batch_normalization/moving_meanPitem_dnn_layer/item_dnn_layer/item_dnn_0/batch_normalization/AssignMovingAvg/mul*[
_classQ
OMloc:@item_dnn_layer/item_dnn_layer/item_dnn_0/batch_normalization/moving_mean*
_output_shapes	
:*
T0
ú
Titem_dnn_layer/item_dnn_layer/item_dnn_0/batch_normalization/AssignMovingAvg_1/decayConst*_
_classU
SQloc:@item_dnn_layer/item_dnn_layer/item_dnn_0/batch_normalization/moving_variance*
dtype0*
_output_shapes
: *
valueB
 *
×#<

Ritem_dnn_layer/item_dnn_layer/item_dnn_0/batch_normalization/AssignMovingAvg_1/subSubQitem_dnn_layer/item_dnn_layer/item_dnn_0/batch_normalization/moving_variance/readNitem_dnn_layer/item_dnn_layer/item_dnn_0/batch_normalization/moments/Squeeze_1*
T0*_
_classU
SQloc:@item_dnn_layer/item_dnn_layer/item_dnn_0/batch_normalization/moving_variance*
_output_shapes	
:

Ritem_dnn_layer/item_dnn_layer/item_dnn_0/batch_normalization/AssignMovingAvg_1/mulMulRitem_dnn_layer/item_dnn_layer/item_dnn_0/batch_normalization/AssignMovingAvg_1/subTitem_dnn_layer/item_dnn_layer/item_dnn_0/batch_normalization/AssignMovingAvg_1/decay*
T0*
_output_shapes	
:*_
_classU
SQloc:@item_dnn_layer/item_dnn_layer/item_dnn_0/batch_normalization/moving_variance

Nitem_dnn_layer/item_dnn_layer/item_dnn_0/batch_normalization/AssignMovingAvg_1	AssignSubLitem_dnn_layer/item_dnn_layer/item_dnn_0/batch_normalization/moving_varianceRitem_dnn_layer/item_dnn_layer/item_dnn_0/batch_normalization/AssignMovingAvg_1/mul*_
_classU
SQloc:@item_dnn_layer/item_dnn_layer/item_dnn_0/batch_normalization/moving_variance*
T0*
_output_shapes	
:
Â
Aitem_dnn_layer/item_dnn_layer/item_dnn_0/batch_normalization/betaIdentityMitem_dnn_layer/item_dnn_layer/item_dnn_0/batch_normalization/beta/part_0/read*
_output_shapes	
:*
T0
Ä
Bitem_dnn_layer/item_dnn_layer/item_dnn_0/batch_normalization/gammaIdentityNitem_dnn_layer/item_dnn_layer/item_dnn_0/batch_normalization/gamma/part_0/read*
_output_shapes	
:*
T0

Litem_dnn_layer/item_dnn_layer/item_dnn_0/batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
valueB
 *o:*
dtype0

Jitem_dnn_layer/item_dnn_layer/item_dnn_0/batch_normalization/batchnorm/addAddV2Nitem_dnn_layer/item_dnn_layer/item_dnn_0/batch_normalization/moments/Squeeze_1Litem_dnn_layer/item_dnn_layer/item_dnn_0/batch_normalization/batchnorm/add/y*
_output_shapes	
:*
T0
Ç
Litem_dnn_layer/item_dnn_layer/item_dnn_0/batch_normalization/batchnorm/RsqrtRsqrtJitem_dnn_layer/item_dnn_layer/item_dnn_0/batch_normalization/batchnorm/add*
T0*
_output_shapes	
:

Jitem_dnn_layer/item_dnn_layer/item_dnn_0/batch_normalization/batchnorm/mulMulLitem_dnn_layer/item_dnn_layer/item_dnn_0/batch_normalization/batchnorm/RsqrtBitem_dnn_layer/item_dnn_layer/item_dnn_0/batch_normalization/gamma*
_output_shapes	
:*
T0

Litem_dnn_layer/item_dnn_layer/item_dnn_0/batch_normalization/batchnorm/mul_1Mul6item_dnn_layer/item_dnn_layer/item_dnn_0/dense/BiasAddJitem_dnn_layer/item_dnn_layer/item_dnn_0/batch_normalization/batchnorm/mul*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

Litem_dnn_layer/item_dnn_layer/item_dnn_0/batch_normalization/batchnorm/mul_2MulLitem_dnn_layer/item_dnn_layer/item_dnn_0/batch_normalization/moments/SqueezeJitem_dnn_layer/item_dnn_layer/item_dnn_0/batch_normalization/batchnorm/mul*
_output_shapes	
:*
T0

Jitem_dnn_layer/item_dnn_layer/item_dnn_0/batch_normalization/batchnorm/subSubAitem_dnn_layer/item_dnn_layer/item_dnn_0/batch_normalization/betaLitem_dnn_layer/item_dnn_layer/item_dnn_0/batch_normalization/batchnorm/mul_2*
_output_shapes	
:*
T0
˘
Litem_dnn_layer/item_dnn_layer/item_dnn_0/batch_normalization/batchnorm/add_1AddV2Litem_dnn_layer/item_dnn_layer/item_dnn_0/batch_normalization/batchnorm/mul_1Jitem_dnn_layer/item_dnn_layer/item_dnn_0/batch_normalization/batchnorm/sub*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ś
-item_dnn_layer/item_dnn_layer/item_dnn_0/ReluReluLitem_dnn_layer/item_dnn_layer/item_dnn_0/batch_normalization/batchnorm/add_1*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ł
;item_dnn_layer/item_dnn_layer/item_dnn_0/zero_fraction/SizeSize-item_dnn_layer/item_dnn_layer/item_dnn_0/Relu*
T0*
_output_shapes
: *
out_type0	

Bitem_dnn_layer/item_dnn_layer/item_dnn_0/zero_fraction/LessEqual/yConst*
_output_shapes
: *
dtype0	*
valueB	 R˙˙˙˙
ď
@item_dnn_layer/item_dnn_layer/item_dnn_0/zero_fraction/LessEqual	LessEqual;item_dnn_layer/item_dnn_layer/item_dnn_0/zero_fraction/SizeBitem_dnn_layer/item_dnn_layer/item_dnn_0/zero_fraction/LessEqual/y*
_output_shapes
: *
T0	
ó
Bitem_dnn_layer/item_dnn_layer/item_dnn_0/zero_fraction/cond/SwitchSwitch@item_dnn_layer/item_dnn_layer/item_dnn_0/zero_fraction/LessEqual@item_dnn_layer/item_dnn_layer/item_dnn_0/zero_fraction/LessEqual*
T0
*
_output_shapes
: : 
ˇ
Ditem_dnn_layer/item_dnn_layer/item_dnn_0/zero_fraction/cond/switch_tIdentityDitem_dnn_layer/item_dnn_layer/item_dnn_0/zero_fraction/cond/Switch:1*
T0
*
_output_shapes
: 
ľ
Ditem_dnn_layer/item_dnn_layer/item_dnn_0/zero_fraction/cond/switch_fIdentityBitem_dnn_layer/item_dnn_layer/item_dnn_0/zero_fraction/cond/Switch*
T0
*
_output_shapes
: 
˛
Citem_dnn_layer/item_dnn_layer/item_dnn_0/zero_fraction/cond/pred_idIdentity@item_dnn_layer/item_dnn_layer/item_dnn_0/zero_fraction/LessEqual*
_output_shapes
: *
T0

Ű
Oitem_dnn_layer/item_dnn_layer/item_dnn_0/zero_fraction/cond/count_nonzero/zerosConstE^item_dnn_layer/item_dnn_layer/item_dnn_0/zero_fraction/cond/switch_t*
dtype0*
valueB
 *    *
_output_shapes
: 
ż
Ritem_dnn_layer/item_dnn_layer/item_dnn_0/zero_fraction/cond/count_nonzero/NotEqualNotEqual[item_dnn_layer/item_dnn_layer/item_dnn_0/zero_fraction/cond/count_nonzero/NotEqual/Switch:1Oitem_dnn_layer/item_dnn_layer/item_dnn_0/zero_fraction/cond/count_nonzero/zeros*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
ŕ
Yitem_dnn_layer/item_dnn_layer/item_dnn_0/zero_fraction/cond/count_nonzero/NotEqual/SwitchSwitch-item_dnn_layer/item_dnn_layer/item_dnn_0/ReluCitem_dnn_layer/item_dnn_layer/item_dnn_0/zero_fraction/cond/pred_id*
T0*@
_class6
42loc:@item_dnn_layer/item_dnn_layer/item_dnn_0/Relu*<
_output_shapes*
(:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
ě
Nitem_dnn_layer/item_dnn_layer/item_dnn_0/zero_fraction/cond/count_nonzero/CastCastRitem_dnn_layer/item_dnn_layer/item_dnn_0/zero_fraction/cond/count_nonzero/NotEqual*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*

SrcT0
*

DstT0
ç
Oitem_dnn_layer/item_dnn_layer/item_dnn_0/zero_fraction/cond/count_nonzero/ConstConstE^item_dnn_layer/item_dnn_layer/item_dnn_0/zero_fraction/cond/switch_t*
_output_shapes
:*
valueB"       *
dtype0
 
Witem_dnn_layer/item_dnn_layer/item_dnn_0/zero_fraction/cond/count_nonzero/nonzero_countSumNitem_dnn_layer/item_dnn_layer/item_dnn_0/zero_fraction/cond/count_nonzero/CastOitem_dnn_layer/item_dnn_layer/item_dnn_0/zero_fraction/cond/count_nonzero/Const*
_output_shapes
: *
T0
Ń
@item_dnn_layer/item_dnn_layer/item_dnn_0/zero_fraction/cond/CastCastWitem_dnn_layer/item_dnn_layer/item_dnn_0/zero_fraction/cond/count_nonzero/nonzero_count*
_output_shapes
: *

SrcT0*

DstT0	
Ý
Qitem_dnn_layer/item_dnn_layer/item_dnn_0/zero_fraction/cond/count_nonzero_1/zerosConstE^item_dnn_layer/item_dnn_layer/item_dnn_0/zero_fraction/cond/switch_f*
_output_shapes
: *
valueB
 *    *
dtype0
Ă
Titem_dnn_layer/item_dnn_layer/item_dnn_0/zero_fraction/cond/count_nonzero_1/NotEqualNotEqual[item_dnn_layer/item_dnn_layer/item_dnn_0/zero_fraction/cond/count_nonzero_1/NotEqual/SwitchQitem_dnn_layer/item_dnn_layer/item_dnn_0/zero_fraction/cond/count_nonzero_1/zeros*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
â
[item_dnn_layer/item_dnn_layer/item_dnn_0/zero_fraction/cond/count_nonzero_1/NotEqual/SwitchSwitch-item_dnn_layer/item_dnn_layer/item_dnn_0/ReluCitem_dnn_layer/item_dnn_layer/item_dnn_0/zero_fraction/cond/pred_id*<
_output_shapes*
(:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*@
_class6
42loc:@item_dnn_layer/item_dnn_layer/item_dnn_0/Relu*
T0
đ
Pitem_dnn_layer/item_dnn_layer/item_dnn_0/zero_fraction/cond/count_nonzero_1/CastCastTitem_dnn_layer/item_dnn_layer/item_dnn_0/zero_fraction/cond/count_nonzero_1/NotEqual*

SrcT0
*

DstT0	*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
é
Qitem_dnn_layer/item_dnn_layer/item_dnn_0/zero_fraction/cond/count_nonzero_1/ConstConstE^item_dnn_layer/item_dnn_layer/item_dnn_0/zero_fraction/cond/switch_f*
_output_shapes
:*
dtype0*
valueB"       
Ś
Yitem_dnn_layer/item_dnn_layer/item_dnn_0/zero_fraction/cond/count_nonzero_1/nonzero_countSumPitem_dnn_layer/item_dnn_layer/item_dnn_0/zero_fraction/cond/count_nonzero_1/CastQitem_dnn_layer/item_dnn_layer/item_dnn_0/zero_fraction/cond/count_nonzero_1/Const*
T0	*
_output_shapes
: 

Aitem_dnn_layer/item_dnn_layer/item_dnn_0/zero_fraction/cond/MergeMergeYitem_dnn_layer/item_dnn_layer/item_dnn_0/zero_fraction/cond/count_nonzero_1/nonzero_count@item_dnn_layer/item_dnn_layer/item_dnn_0/zero_fraction/cond/Cast*
N*
T0	*
_output_shapes
: : 
ő
Mitem_dnn_layer/item_dnn_layer/item_dnn_0/zero_fraction/counts_to_fraction/subSub;item_dnn_layer/item_dnn_layer/item_dnn_0/zero_fraction/SizeAitem_dnn_layer/item_dnn_layer/item_dnn_0/zero_fraction/cond/Merge*
_output_shapes
: *
T0	
Ő
Nitem_dnn_layer/item_dnn_layer/item_dnn_0/zero_fraction/counts_to_fraction/CastCastMitem_dnn_layer/item_dnn_layer/item_dnn_0/zero_fraction/counts_to_fraction/sub*

SrcT0	*

DstT0*
_output_shapes
: 
Ĺ
Pitem_dnn_layer/item_dnn_layer/item_dnn_0/zero_fraction/counts_to_fraction/Cast_1Cast;item_dnn_layer/item_dnn_layer/item_dnn_0/zero_fraction/Size*
_output_shapes
: *

DstT0*

SrcT0	

Qitem_dnn_layer/item_dnn_layer/item_dnn_0/zero_fraction/counts_to_fraction/truedivRealDivNitem_dnn_layer/item_dnn_layer/item_dnn_0/zero_fraction/counts_to_fraction/CastPitem_dnn_layer/item_dnn_layer/item_dnn_0/zero_fraction/counts_to_fraction/Cast_1*
T0*
_output_shapes
: 
ż
?item_dnn_layer/item_dnn_layer/item_dnn_0/zero_fraction/fractionIdentityQitem_dnn_layer/item_dnn_layer/item_dnn_0/zero_fraction/counts_to_fraction/truediv*
T0*
_output_shapes
: 

nitem_dnn_layer/item_dnn_layer/item_dnn_0/item_dnn_layer/item_dnn_layer/item_dnn_0/fraction_of_zero_values/tagsConst*
dtype0*z
valueqBo Biitem_dnn_layer/item_dnn_layer/item_dnn_0/item_dnn_layer/item_dnn_layer/item_dnn_0/fraction_of_zero_values*
_output_shapes
: 
Ě
iitem_dnn_layer/item_dnn_layer/item_dnn_0/item_dnn_layer/item_dnn_layer/item_dnn_0/fraction_of_zero_valuesScalarSummarynitem_dnn_layer/item_dnn_layer/item_dnn_0/item_dnn_layer/item_dnn_layer/item_dnn_0/fraction_of_zero_values/tags?item_dnn_layer/item_dnn_layer/item_dnn_0/zero_fraction/fraction*
T0*
_output_shapes
: 
ý
`item_dnn_layer/item_dnn_layer/item_dnn_0/item_dnn_layer/item_dnn_layer/item_dnn_0/activation/tagConst*
dtype0*
_output_shapes
: *m
valuedBb B\item_dnn_layer/item_dnn_layer/item_dnn_0/item_dnn_layer/item_dnn_layer/item_dnn_0/activation

\item_dnn_layer/item_dnn_layer/item_dnn_0/item_dnn_layer/item_dnn_layer/item_dnn_0/activationHistogramSummary`item_dnn_layer/item_dnn_layer/item_dnn_0/item_dnn_layer/item_dnn_layer/item_dnn_0/activation/tag-item_dnn_layer/item_dnn_layer/item_dnn_0/Relu*
_output_shapes
: 
˙
]item_dnn_layer/item_dnn_layer/item_dnn_1/dense/kernel/part_0/Initializer/random_uniform/shapeConst*
_output_shapes
:*O
_classE
CAloc:@item_dnn_layer/item_dnn_layer/item_dnn_1/dense/kernel/part_0*
valueB"      *
dtype0
ń
[item_dnn_layer/item_dnn_layer/item_dnn_1/dense/kernel/part_0/Initializer/random_uniform/minConst*O
_classE
CAloc:@item_dnn_layer/item_dnn_layer/item_dnn_1/dense/kernel/part_0*
dtype0*
_output_shapes
: *
valueB
 *   ž
ń
[item_dnn_layer/item_dnn_layer/item_dnn_1/dense/kernel/part_0/Initializer/random_uniform/maxConst*
valueB
 *   >*
dtype0*
_output_shapes
: *O
_classE
CAloc:@item_dnn_layer/item_dnn_layer/item_dnn_1/dense/kernel/part_0
Ţ
eitem_dnn_layer/item_dnn_layer/item_dnn_1/dense/kernel/part_0/Initializer/random_uniform/RandomUniformRandomUniform]item_dnn_layer/item_dnn_layer/item_dnn_1/dense/kernel/part_0/Initializer/random_uniform/shape*O
_classE
CAloc:@item_dnn_layer/item_dnn_layer/item_dnn_1/dense/kernel/part_0*
dtype0*
T0* 
_output_shapes
:


[item_dnn_layer/item_dnn_layer/item_dnn_1/dense/kernel/part_0/Initializer/random_uniform/subSub[item_dnn_layer/item_dnn_layer/item_dnn_1/dense/kernel/part_0/Initializer/random_uniform/max[item_dnn_layer/item_dnn_layer/item_dnn_1/dense/kernel/part_0/Initializer/random_uniform/min*O
_classE
CAloc:@item_dnn_layer/item_dnn_layer/item_dnn_1/dense/kernel/part_0*
T0*
_output_shapes
: 
˘
[item_dnn_layer/item_dnn_layer/item_dnn_1/dense/kernel/part_0/Initializer/random_uniform/mulMuleitem_dnn_layer/item_dnn_layer/item_dnn_1/dense/kernel/part_0/Initializer/random_uniform/RandomUniform[item_dnn_layer/item_dnn_layer/item_dnn_1/dense/kernel/part_0/Initializer/random_uniform/sub* 
_output_shapes
:
*
T0*O
_classE
CAloc:@item_dnn_layer/item_dnn_layer/item_dnn_1/dense/kernel/part_0

Witem_dnn_layer/item_dnn_layer/item_dnn_1/dense/kernel/part_0/Initializer/random_uniformAdd[item_dnn_layer/item_dnn_layer/item_dnn_1/dense/kernel/part_0/Initializer/random_uniform/mul[item_dnn_layer/item_dnn_layer/item_dnn_1/dense/kernel/part_0/Initializer/random_uniform/min* 
_output_shapes
:
*O
_classE
CAloc:@item_dnn_layer/item_dnn_layer/item_dnn_1/dense/kernel/part_0*
T0
á
<item_dnn_layer/item_dnn_layer/item_dnn_1/dense/kernel/part_0
VariableV2*
shape:
*O
_classE
CAloc:@item_dnn_layer/item_dnn_layer/item_dnn_1/dense/kernel/part_0*
dtype0* 
_output_shapes
:

ŕ
Citem_dnn_layer/item_dnn_layer/item_dnn_1/dense/kernel/part_0/AssignAssign<item_dnn_layer/item_dnn_layer/item_dnn_1/dense/kernel/part_0Witem_dnn_layer/item_dnn_layer/item_dnn_1/dense/kernel/part_0/Initializer/random_uniform*
T0* 
_output_shapes
:
*O
_classE
CAloc:@item_dnn_layer/item_dnn_layer/item_dnn_1/dense/kernel/part_0

Aitem_dnn_layer/item_dnn_layer/item_dnn_1/dense/kernel/part_0/readIdentity<item_dnn_layer/item_dnn_layer/item_dnn_1/dense/kernel/part_0*
T0* 
_output_shapes
:
*O
_classE
CAloc:@item_dnn_layer/item_dnn_layer/item_dnn_1/dense/kernel/part_0
ę
Litem_dnn_layer/item_dnn_layer/item_dnn_1/dense/bias/part_0/Initializer/zerosConst*
valueB*    *M
_classC
A?loc:@item_dnn_layer/item_dnn_layer/item_dnn_1/dense/bias/part_0*
_output_shapes	
:*
dtype0
Ó
:item_dnn_layer/item_dnn_layer/item_dnn_1/dense/bias/part_0
VariableV2*
dtype0*
_output_shapes	
:*
shape:*M
_classC
A?loc:@item_dnn_layer/item_dnn_layer/item_dnn_1/dense/bias/part_0
Ę
Aitem_dnn_layer/item_dnn_layer/item_dnn_1/dense/bias/part_0/AssignAssign:item_dnn_layer/item_dnn_layer/item_dnn_1/dense/bias/part_0Litem_dnn_layer/item_dnn_layer/item_dnn_1/dense/bias/part_0/Initializer/zeros*M
_classC
A?loc:@item_dnn_layer/item_dnn_layer/item_dnn_1/dense/bias/part_0*
_output_shapes	
:*
T0
ü
?item_dnn_layer/item_dnn_layer/item_dnn_1/dense/bias/part_0/readIdentity:item_dnn_layer/item_dnn_layer/item_dnn_1/dense/bias/part_0*M
_classC
A?loc:@item_dnn_layer/item_dnn_layer/item_dnn_1/dense/bias/part_0*
_output_shapes	
:*
T0
Ż
5item_dnn_layer/item_dnn_layer/item_dnn_1/dense/kernelIdentityAitem_dnn_layer/item_dnn_layer/item_dnn_1/dense/kernel/part_0/read*
T0* 
_output_shapes
:

Ř
5item_dnn_layer/item_dnn_layer/item_dnn_1/dense/MatMulMatMul-item_dnn_layer/item_dnn_layer/item_dnn_0/Relu5item_dnn_layer/item_dnn_layer/item_dnn_1/dense/kernel*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ś
3item_dnn_layer/item_dnn_layer/item_dnn_1/dense/biasIdentity?item_dnn_layer/item_dnn_layer/item_dnn_1/dense/bias/part_0/read*
T0*
_output_shapes	
:
ŕ
6item_dnn_layer/item_dnn_layer/item_dnn_1/dense/BiasAddBiasAdd5item_dnn_layer/item_dnn_layer/item_dnn_1/dense/MatMul3item_dnn_layer/item_dnn_layer/item_dnn_1/dense/bias*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

Zitem_dnn_layer/item_dnn_layer/item_dnn_1/batch_normalization/gamma/part_0/Initializer/onesConst*
dtype0*
valueB*  ?*
_output_shapes	
:*\
_classR
PNloc:@item_dnn_layer/item_dnn_layer/item_dnn_1/batch_normalization/gamma/part_0
ń
Iitem_dnn_layer/item_dnn_layer/item_dnn_1/batch_normalization/gamma/part_0
VariableV2*
shape:*\
_classR
PNloc:@item_dnn_layer/item_dnn_layer/item_dnn_1/batch_normalization/gamma/part_0*
_output_shapes	
:*
dtype0

Pitem_dnn_layer/item_dnn_layer/item_dnn_1/batch_normalization/gamma/part_0/AssignAssignIitem_dnn_layer/item_dnn_layer/item_dnn_1/batch_normalization/gamma/part_0Zitem_dnn_layer/item_dnn_layer/item_dnn_1/batch_normalization/gamma/part_0/Initializer/ones*
T0*\
_classR
PNloc:@item_dnn_layer/item_dnn_layer/item_dnn_1/batch_normalization/gamma/part_0*
_output_shapes	
:
Š
Nitem_dnn_layer/item_dnn_layer/item_dnn_1/batch_normalization/gamma/part_0/readIdentityIitem_dnn_layer/item_dnn_layer/item_dnn_1/batch_normalization/gamma/part_0*
T0*\
_classR
PNloc:@item_dnn_layer/item_dnn_layer/item_dnn_1/batch_normalization/gamma/part_0*
_output_shapes	
:

Zitem_dnn_layer/item_dnn_layer/item_dnn_1/batch_normalization/beta/part_0/Initializer/zerosConst*
valueB*    *
dtype0*
_output_shapes	
:*[
_classQ
OMloc:@item_dnn_layer/item_dnn_layer/item_dnn_1/batch_normalization/beta/part_0
ď
Hitem_dnn_layer/item_dnn_layer/item_dnn_1/batch_normalization/beta/part_0
VariableV2*
_output_shapes	
:*
shape:*[
_classQ
OMloc:@item_dnn_layer/item_dnn_layer/item_dnn_1/batch_normalization/beta/part_0*
dtype0

Oitem_dnn_layer/item_dnn_layer/item_dnn_1/batch_normalization/beta/part_0/AssignAssignHitem_dnn_layer/item_dnn_layer/item_dnn_1/batch_normalization/beta/part_0Zitem_dnn_layer/item_dnn_layer/item_dnn_1/batch_normalization/beta/part_0/Initializer/zeros*
_output_shapes	
:*
T0*[
_classQ
OMloc:@item_dnn_layer/item_dnn_layer/item_dnn_1/batch_normalization/beta/part_0
Ś
Mitem_dnn_layer/item_dnn_layer/item_dnn_1/batch_normalization/beta/part_0/readIdentityHitem_dnn_layer/item_dnn_layer/item_dnn_1/batch_normalization/beta/part_0*[
_classQ
OMloc:@item_dnn_layer/item_dnn_layer/item_dnn_1/batch_normalization/beta/part_0*
_output_shapes	
:*
T0

Zitem_dnn_layer/item_dnn_layer/item_dnn_1/batch_normalization/moving_mean/Initializer/zerosConst*
dtype0*[
_classQ
OMloc:@item_dnn_layer/item_dnn_layer/item_dnn_1/batch_normalization/moving_mean*
_output_shapes	
:*
valueB*    
ď
Hitem_dnn_layer/item_dnn_layer/item_dnn_1/batch_normalization/moving_mean
VariableV2*
_output_shapes	
:*[
_classQ
OMloc:@item_dnn_layer/item_dnn_layer/item_dnn_1/batch_normalization/moving_mean*
shape:*
dtype0

Oitem_dnn_layer/item_dnn_layer/item_dnn_1/batch_normalization/moving_mean/AssignAssignHitem_dnn_layer/item_dnn_layer/item_dnn_1/batch_normalization/moving_meanZitem_dnn_layer/item_dnn_layer/item_dnn_1/batch_normalization/moving_mean/Initializer/zeros*
_output_shapes	
:*
T0*[
_classQ
OMloc:@item_dnn_layer/item_dnn_layer/item_dnn_1/batch_normalization/moving_mean
Ś
Mitem_dnn_layer/item_dnn_layer/item_dnn_1/batch_normalization/moving_mean/readIdentityHitem_dnn_layer/item_dnn_layer/item_dnn_1/batch_normalization/moving_mean*
T0*
_output_shapes	
:*[
_classQ
OMloc:@item_dnn_layer/item_dnn_layer/item_dnn_1/batch_normalization/moving_mean

]item_dnn_layer/item_dnn_layer/item_dnn_1/batch_normalization/moving_variance/Initializer/onesConst*
dtype0*
_output_shapes	
:*
valueB*  ?*_
_classU
SQloc:@item_dnn_layer/item_dnn_layer/item_dnn_1/batch_normalization/moving_variance
÷
Litem_dnn_layer/item_dnn_layer/item_dnn_1/batch_normalization/moving_variance
VariableV2*
shape:*
dtype0*_
_classU
SQloc:@item_dnn_layer/item_dnn_layer/item_dnn_1/batch_normalization/moving_variance*
_output_shapes	
:

Sitem_dnn_layer/item_dnn_layer/item_dnn_1/batch_normalization/moving_variance/AssignAssignLitem_dnn_layer/item_dnn_layer/item_dnn_1/batch_normalization/moving_variance]item_dnn_layer/item_dnn_layer/item_dnn_1/batch_normalization/moving_variance/Initializer/ones*
_output_shapes	
:*_
_classU
SQloc:@item_dnn_layer/item_dnn_layer/item_dnn_1/batch_normalization/moving_variance*
T0
˛
Qitem_dnn_layer/item_dnn_layer/item_dnn_1/batch_normalization/moving_variance/readIdentityLitem_dnn_layer/item_dnn_layer/item_dnn_1/batch_normalization/moving_variance*
_output_shapes	
:*
T0*_
_classU
SQloc:@item_dnn_layer/item_dnn_layer/item_dnn_1/batch_normalization/moving_variance
Ľ
[item_dnn_layer/item_dnn_layer/item_dnn_1/batch_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
valueB: *
dtype0
Ą
Iitem_dnn_layer/item_dnn_layer/item_dnn_1/batch_normalization/moments/meanMean6item_dnn_layer/item_dnn_layer/item_dnn_1/dense/BiasAdd[item_dnn_layer/item_dnn_layer/item_dnn_1/batch_normalization/moments/mean/reduction_indices*
T0*
	keep_dims(*
_output_shapes
:	
Ö
Qitem_dnn_layer/item_dnn_layer/item_dnn_1/batch_normalization/moments/StopGradientStopGradientIitem_dnn_layer/item_dnn_layer/item_dnn_1/batch_normalization/moments/mean*
_output_shapes
:	*
T0
Š
Vitem_dnn_layer/item_dnn_layer/item_dnn_1/batch_normalization/moments/SquaredDifferenceSquaredDifference6item_dnn_layer/item_dnn_layer/item_dnn_1/dense/BiasAddQitem_dnn_layer/item_dnn_layer/item_dnn_1/batch_normalization/moments/StopGradient*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Š
_item_dnn_layer/item_dnn_layer/item_dnn_1/batch_normalization/moments/variance/reduction_indicesConst*
dtype0*
_output_shapes
:*
valueB: 
É
Mitem_dnn_layer/item_dnn_layer/item_dnn_1/batch_normalization/moments/varianceMeanVitem_dnn_layer/item_dnn_layer/item_dnn_1/batch_normalization/moments/SquaredDifference_item_dnn_layer/item_dnn_layer/item_dnn_1/batch_normalization/moments/variance/reduction_indices*
_output_shapes
:	*
T0*
	keep_dims(
ß
Litem_dnn_layer/item_dnn_layer/item_dnn_1/batch_normalization/moments/SqueezeSqueezeIitem_dnn_layer/item_dnn_layer/item_dnn_1/batch_normalization/moments/mean*
T0*
_output_shapes	
:*
squeeze_dims
 
ĺ
Nitem_dnn_layer/item_dnn_layer/item_dnn_1/batch_normalization/moments/Squeeze_1SqueezeMitem_dnn_layer/item_dnn_layer/item_dnn_1/batch_normalization/moments/variance*
T0*
_output_shapes	
:*
squeeze_dims
 
ô
Ritem_dnn_layer/item_dnn_layer/item_dnn_1/batch_normalization/AssignMovingAvg/decayConst*
valueB
 *
×#<*[
_classQ
OMloc:@item_dnn_layer/item_dnn_layer/item_dnn_1/batch_normalization/moving_mean*
dtype0*
_output_shapes
: 
÷
Pitem_dnn_layer/item_dnn_layer/item_dnn_1/batch_normalization/AssignMovingAvg/subSubMitem_dnn_layer/item_dnn_layer/item_dnn_1/batch_normalization/moving_mean/readLitem_dnn_layer/item_dnn_layer/item_dnn_1/batch_normalization/moments/Squeeze*
_output_shapes	
:*[
_classQ
OMloc:@item_dnn_layer/item_dnn_layer/item_dnn_1/batch_normalization/moving_mean*
T0

Pitem_dnn_layer/item_dnn_layer/item_dnn_1/batch_normalization/AssignMovingAvg/mulMulPitem_dnn_layer/item_dnn_layer/item_dnn_1/batch_normalization/AssignMovingAvg/subRitem_dnn_layer/item_dnn_layer/item_dnn_1/batch_normalization/AssignMovingAvg/decay*
T0*[
_classQ
OMloc:@item_dnn_layer/item_dnn_layer/item_dnn_1/batch_normalization/moving_mean*
_output_shapes	
:
ř
Litem_dnn_layer/item_dnn_layer/item_dnn_1/batch_normalization/AssignMovingAvg	AssignSubHitem_dnn_layer/item_dnn_layer/item_dnn_1/batch_normalization/moving_meanPitem_dnn_layer/item_dnn_layer/item_dnn_1/batch_normalization/AssignMovingAvg/mul*[
_classQ
OMloc:@item_dnn_layer/item_dnn_layer/item_dnn_1/batch_normalization/moving_mean*
_output_shapes	
:*
T0
ú
Titem_dnn_layer/item_dnn_layer/item_dnn_1/batch_normalization/AssignMovingAvg_1/decayConst*
_output_shapes
: *
valueB
 *
×#<*_
_classU
SQloc:@item_dnn_layer/item_dnn_layer/item_dnn_1/batch_normalization/moving_variance*
dtype0

Ritem_dnn_layer/item_dnn_layer/item_dnn_1/batch_normalization/AssignMovingAvg_1/subSubQitem_dnn_layer/item_dnn_layer/item_dnn_1/batch_normalization/moving_variance/readNitem_dnn_layer/item_dnn_layer/item_dnn_1/batch_normalization/moments/Squeeze_1*
_output_shapes	
:*
T0*_
_classU
SQloc:@item_dnn_layer/item_dnn_layer/item_dnn_1/batch_normalization/moving_variance

Ritem_dnn_layer/item_dnn_layer/item_dnn_1/batch_normalization/AssignMovingAvg_1/mulMulRitem_dnn_layer/item_dnn_layer/item_dnn_1/batch_normalization/AssignMovingAvg_1/subTitem_dnn_layer/item_dnn_layer/item_dnn_1/batch_normalization/AssignMovingAvg_1/decay*
T0*
_output_shapes	
:*_
_classU
SQloc:@item_dnn_layer/item_dnn_layer/item_dnn_1/batch_normalization/moving_variance

Nitem_dnn_layer/item_dnn_layer/item_dnn_1/batch_normalization/AssignMovingAvg_1	AssignSubLitem_dnn_layer/item_dnn_layer/item_dnn_1/batch_normalization/moving_varianceRitem_dnn_layer/item_dnn_layer/item_dnn_1/batch_normalization/AssignMovingAvg_1/mul*
_output_shapes	
:*
T0*_
_classU
SQloc:@item_dnn_layer/item_dnn_layer/item_dnn_1/batch_normalization/moving_variance
Â
Aitem_dnn_layer/item_dnn_layer/item_dnn_1/batch_normalization/betaIdentityMitem_dnn_layer/item_dnn_layer/item_dnn_1/batch_normalization/beta/part_0/read*
T0*
_output_shapes	
:
Ä
Bitem_dnn_layer/item_dnn_layer/item_dnn_1/batch_normalization/gammaIdentityNitem_dnn_layer/item_dnn_layer/item_dnn_1/batch_normalization/gamma/part_0/read*
_output_shapes	
:*
T0

Litem_dnn_layer/item_dnn_layer/item_dnn_1/batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
valueB
 *o:*
dtype0

Jitem_dnn_layer/item_dnn_layer/item_dnn_1/batch_normalization/batchnorm/addAddV2Nitem_dnn_layer/item_dnn_layer/item_dnn_1/batch_normalization/moments/Squeeze_1Litem_dnn_layer/item_dnn_layer/item_dnn_1/batch_normalization/batchnorm/add/y*
T0*
_output_shapes	
:
Ç
Litem_dnn_layer/item_dnn_layer/item_dnn_1/batch_normalization/batchnorm/RsqrtRsqrtJitem_dnn_layer/item_dnn_layer/item_dnn_1/batch_normalization/batchnorm/add*
T0*
_output_shapes	
:

Jitem_dnn_layer/item_dnn_layer/item_dnn_1/batch_normalization/batchnorm/mulMulLitem_dnn_layer/item_dnn_layer/item_dnn_1/batch_normalization/batchnorm/RsqrtBitem_dnn_layer/item_dnn_layer/item_dnn_1/batch_normalization/gamma*
_output_shapes	
:*
T0

Litem_dnn_layer/item_dnn_layer/item_dnn_1/batch_normalization/batchnorm/mul_1Mul6item_dnn_layer/item_dnn_layer/item_dnn_1/dense/BiasAddJitem_dnn_layer/item_dnn_layer/item_dnn_1/batch_normalization/batchnorm/mul*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

Litem_dnn_layer/item_dnn_layer/item_dnn_1/batch_normalization/batchnorm/mul_2MulLitem_dnn_layer/item_dnn_layer/item_dnn_1/batch_normalization/moments/SqueezeJitem_dnn_layer/item_dnn_layer/item_dnn_1/batch_normalization/batchnorm/mul*
T0*
_output_shapes	
:

Jitem_dnn_layer/item_dnn_layer/item_dnn_1/batch_normalization/batchnorm/subSubAitem_dnn_layer/item_dnn_layer/item_dnn_1/batch_normalization/betaLitem_dnn_layer/item_dnn_layer/item_dnn_1/batch_normalization/batchnorm/mul_2*
T0*
_output_shapes	
:
˘
Litem_dnn_layer/item_dnn_layer/item_dnn_1/batch_normalization/batchnorm/add_1AddV2Litem_dnn_layer/item_dnn_layer/item_dnn_1/batch_normalization/batchnorm/mul_1Jitem_dnn_layer/item_dnn_layer/item_dnn_1/batch_normalization/batchnorm/sub*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ś
-item_dnn_layer/item_dnn_layer/item_dnn_1/ReluReluLitem_dnn_layer/item_dnn_layer/item_dnn_1/batch_normalization/batchnorm/add_1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ł
;item_dnn_layer/item_dnn_layer/item_dnn_1/zero_fraction/SizeSize-item_dnn_layer/item_dnn_layer/item_dnn_1/Relu*
out_type0	*
_output_shapes
: *
T0

Bitem_dnn_layer/item_dnn_layer/item_dnn_1/zero_fraction/LessEqual/yConst*
valueB	 R˙˙˙˙*
dtype0	*
_output_shapes
: 
ď
@item_dnn_layer/item_dnn_layer/item_dnn_1/zero_fraction/LessEqual	LessEqual;item_dnn_layer/item_dnn_layer/item_dnn_1/zero_fraction/SizeBitem_dnn_layer/item_dnn_layer/item_dnn_1/zero_fraction/LessEqual/y*
T0	*
_output_shapes
: 
ó
Bitem_dnn_layer/item_dnn_layer/item_dnn_1/zero_fraction/cond/SwitchSwitch@item_dnn_layer/item_dnn_layer/item_dnn_1/zero_fraction/LessEqual@item_dnn_layer/item_dnn_layer/item_dnn_1/zero_fraction/LessEqual*
_output_shapes
: : *
T0

ˇ
Ditem_dnn_layer/item_dnn_layer/item_dnn_1/zero_fraction/cond/switch_tIdentityDitem_dnn_layer/item_dnn_layer/item_dnn_1/zero_fraction/cond/Switch:1*
_output_shapes
: *
T0

ľ
Ditem_dnn_layer/item_dnn_layer/item_dnn_1/zero_fraction/cond/switch_fIdentityBitem_dnn_layer/item_dnn_layer/item_dnn_1/zero_fraction/cond/Switch*
_output_shapes
: *
T0

˛
Citem_dnn_layer/item_dnn_layer/item_dnn_1/zero_fraction/cond/pred_idIdentity@item_dnn_layer/item_dnn_layer/item_dnn_1/zero_fraction/LessEqual*
T0
*
_output_shapes
: 
Ű
Oitem_dnn_layer/item_dnn_layer/item_dnn_1/zero_fraction/cond/count_nonzero/zerosConstE^item_dnn_layer/item_dnn_layer/item_dnn_1/zero_fraction/cond/switch_t*
dtype0*
_output_shapes
: *
valueB
 *    
ż
Ritem_dnn_layer/item_dnn_layer/item_dnn_1/zero_fraction/cond/count_nonzero/NotEqualNotEqual[item_dnn_layer/item_dnn_layer/item_dnn_1/zero_fraction/cond/count_nonzero/NotEqual/Switch:1Oitem_dnn_layer/item_dnn_layer/item_dnn_1/zero_fraction/cond/count_nonzero/zeros*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
ŕ
Yitem_dnn_layer/item_dnn_layer/item_dnn_1/zero_fraction/cond/count_nonzero/NotEqual/SwitchSwitch-item_dnn_layer/item_dnn_layer/item_dnn_1/ReluCitem_dnn_layer/item_dnn_layer/item_dnn_1/zero_fraction/cond/pred_id*<
_output_shapes*
(:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*@
_class6
42loc:@item_dnn_layer/item_dnn_layer/item_dnn_1/Relu*
T0
ě
Nitem_dnn_layer/item_dnn_layer/item_dnn_1/zero_fraction/cond/count_nonzero/CastCastRitem_dnn_layer/item_dnn_layer/item_dnn_1/zero_fraction/cond/count_nonzero/NotEqual*

SrcT0
*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*

DstT0
ç
Oitem_dnn_layer/item_dnn_layer/item_dnn_1/zero_fraction/cond/count_nonzero/ConstConstE^item_dnn_layer/item_dnn_layer/item_dnn_1/zero_fraction/cond/switch_t*
dtype0*
_output_shapes
:*
valueB"       
 
Witem_dnn_layer/item_dnn_layer/item_dnn_1/zero_fraction/cond/count_nonzero/nonzero_countSumNitem_dnn_layer/item_dnn_layer/item_dnn_1/zero_fraction/cond/count_nonzero/CastOitem_dnn_layer/item_dnn_layer/item_dnn_1/zero_fraction/cond/count_nonzero/Const*
T0*
_output_shapes
: 
Ń
@item_dnn_layer/item_dnn_layer/item_dnn_1/zero_fraction/cond/CastCastWitem_dnn_layer/item_dnn_layer/item_dnn_1/zero_fraction/cond/count_nonzero/nonzero_count*
_output_shapes
: *

SrcT0*

DstT0	
Ý
Qitem_dnn_layer/item_dnn_layer/item_dnn_1/zero_fraction/cond/count_nonzero_1/zerosConstE^item_dnn_layer/item_dnn_layer/item_dnn_1/zero_fraction/cond/switch_f*
_output_shapes
: *
valueB
 *    *
dtype0
Ă
Titem_dnn_layer/item_dnn_layer/item_dnn_1/zero_fraction/cond/count_nonzero_1/NotEqualNotEqual[item_dnn_layer/item_dnn_layer/item_dnn_1/zero_fraction/cond/count_nonzero_1/NotEqual/SwitchQitem_dnn_layer/item_dnn_layer/item_dnn_1/zero_fraction/cond/count_nonzero_1/zeros*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
â
[item_dnn_layer/item_dnn_layer/item_dnn_1/zero_fraction/cond/count_nonzero_1/NotEqual/SwitchSwitch-item_dnn_layer/item_dnn_layer/item_dnn_1/ReluCitem_dnn_layer/item_dnn_layer/item_dnn_1/zero_fraction/cond/pred_id*<
_output_shapes*
(:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*@
_class6
42loc:@item_dnn_layer/item_dnn_layer/item_dnn_1/Relu*
T0
đ
Pitem_dnn_layer/item_dnn_layer/item_dnn_1/zero_fraction/cond/count_nonzero_1/CastCastTitem_dnn_layer/item_dnn_layer/item_dnn_1/zero_fraction/cond/count_nonzero_1/NotEqual*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*

SrcT0
*

DstT0	
é
Qitem_dnn_layer/item_dnn_layer/item_dnn_1/zero_fraction/cond/count_nonzero_1/ConstConstE^item_dnn_layer/item_dnn_layer/item_dnn_1/zero_fraction/cond/switch_f*
valueB"       *
dtype0*
_output_shapes
:
Ś
Yitem_dnn_layer/item_dnn_layer/item_dnn_1/zero_fraction/cond/count_nonzero_1/nonzero_countSumPitem_dnn_layer/item_dnn_layer/item_dnn_1/zero_fraction/cond/count_nonzero_1/CastQitem_dnn_layer/item_dnn_layer/item_dnn_1/zero_fraction/cond/count_nonzero_1/Const*
T0	*
_output_shapes
: 

Aitem_dnn_layer/item_dnn_layer/item_dnn_1/zero_fraction/cond/MergeMergeYitem_dnn_layer/item_dnn_layer/item_dnn_1/zero_fraction/cond/count_nonzero_1/nonzero_count@item_dnn_layer/item_dnn_layer/item_dnn_1/zero_fraction/cond/Cast*
N*
T0	*
_output_shapes
: : 
ő
Mitem_dnn_layer/item_dnn_layer/item_dnn_1/zero_fraction/counts_to_fraction/subSub;item_dnn_layer/item_dnn_layer/item_dnn_1/zero_fraction/SizeAitem_dnn_layer/item_dnn_layer/item_dnn_1/zero_fraction/cond/Merge*
_output_shapes
: *
T0	
Ő
Nitem_dnn_layer/item_dnn_layer/item_dnn_1/zero_fraction/counts_to_fraction/CastCastMitem_dnn_layer/item_dnn_layer/item_dnn_1/zero_fraction/counts_to_fraction/sub*
_output_shapes
: *

DstT0*

SrcT0	
Ĺ
Pitem_dnn_layer/item_dnn_layer/item_dnn_1/zero_fraction/counts_to_fraction/Cast_1Cast;item_dnn_layer/item_dnn_layer/item_dnn_1/zero_fraction/Size*

DstT0*
_output_shapes
: *

SrcT0	

Qitem_dnn_layer/item_dnn_layer/item_dnn_1/zero_fraction/counts_to_fraction/truedivRealDivNitem_dnn_layer/item_dnn_layer/item_dnn_1/zero_fraction/counts_to_fraction/CastPitem_dnn_layer/item_dnn_layer/item_dnn_1/zero_fraction/counts_to_fraction/Cast_1*
T0*
_output_shapes
: 
ż
?item_dnn_layer/item_dnn_layer/item_dnn_1/zero_fraction/fractionIdentityQitem_dnn_layer/item_dnn_layer/item_dnn_1/zero_fraction/counts_to_fraction/truediv*
_output_shapes
: *
T0

nitem_dnn_layer/item_dnn_layer/item_dnn_1/item_dnn_layer/item_dnn_layer/item_dnn_1/fraction_of_zero_values/tagsConst*z
valueqBo Biitem_dnn_layer/item_dnn_layer/item_dnn_1/item_dnn_layer/item_dnn_layer/item_dnn_1/fraction_of_zero_values*
_output_shapes
: *
dtype0
Ě
iitem_dnn_layer/item_dnn_layer/item_dnn_1/item_dnn_layer/item_dnn_layer/item_dnn_1/fraction_of_zero_valuesScalarSummarynitem_dnn_layer/item_dnn_layer/item_dnn_1/item_dnn_layer/item_dnn_layer/item_dnn_1/fraction_of_zero_values/tags?item_dnn_layer/item_dnn_layer/item_dnn_1/zero_fraction/fraction*
_output_shapes
: *
T0
ý
`item_dnn_layer/item_dnn_layer/item_dnn_1/item_dnn_layer/item_dnn_layer/item_dnn_1/activation/tagConst*m
valuedBb B\item_dnn_layer/item_dnn_layer/item_dnn_1/item_dnn_layer/item_dnn_layer/item_dnn_1/activation*
_output_shapes
: *
dtype0

\item_dnn_layer/item_dnn_layer/item_dnn_1/item_dnn_layer/item_dnn_layer/item_dnn_1/activationHistogramSummary`item_dnn_layer/item_dnn_layer/item_dnn_1/item_dnn_layer/item_dnn_layer/item_dnn_1/activation/tag-item_dnn_layer/item_dnn_layer/item_dnn_1/Relu*
_output_shapes
: 
˙
]item_dnn_layer/item_dnn_layer/item_dnn_2/dense/kernel/part_0/Initializer/random_uniform/shapeConst*
_output_shapes
:*
dtype0*
valueB"   @   *O
_classE
CAloc:@item_dnn_layer/item_dnn_layer/item_dnn_2/dense/kernel/part_0
ń
[item_dnn_layer/item_dnn_layer/item_dnn_2/dense/kernel/part_0/Initializer/random_uniform/minConst*
valueB
 *ó5ž*
dtype0*
_output_shapes
: *O
_classE
CAloc:@item_dnn_layer/item_dnn_layer/item_dnn_2/dense/kernel/part_0
ń
[item_dnn_layer/item_dnn_layer/item_dnn_2/dense/kernel/part_0/Initializer/random_uniform/maxConst*
_output_shapes
: *O
_classE
CAloc:@item_dnn_layer/item_dnn_layer/item_dnn_2/dense/kernel/part_0*
valueB
 *ó5>*
dtype0
Ý
eitem_dnn_layer/item_dnn_layer/item_dnn_2/dense/kernel/part_0/Initializer/random_uniform/RandomUniformRandomUniform]item_dnn_layer/item_dnn_layer/item_dnn_2/dense/kernel/part_0/Initializer/random_uniform/shape*
dtype0*O
_classE
CAloc:@item_dnn_layer/item_dnn_layer/item_dnn_2/dense/kernel/part_0*
T0*
_output_shapes
:	@

[item_dnn_layer/item_dnn_layer/item_dnn_2/dense/kernel/part_0/Initializer/random_uniform/subSub[item_dnn_layer/item_dnn_layer/item_dnn_2/dense/kernel/part_0/Initializer/random_uniform/max[item_dnn_layer/item_dnn_layer/item_dnn_2/dense/kernel/part_0/Initializer/random_uniform/min*
T0*
_output_shapes
: *O
_classE
CAloc:@item_dnn_layer/item_dnn_layer/item_dnn_2/dense/kernel/part_0
Ą
[item_dnn_layer/item_dnn_layer/item_dnn_2/dense/kernel/part_0/Initializer/random_uniform/mulMuleitem_dnn_layer/item_dnn_layer/item_dnn_2/dense/kernel/part_0/Initializer/random_uniform/RandomUniform[item_dnn_layer/item_dnn_layer/item_dnn_2/dense/kernel/part_0/Initializer/random_uniform/sub*O
_classE
CAloc:@item_dnn_layer/item_dnn_layer/item_dnn_2/dense/kernel/part_0*
T0*
_output_shapes
:	@

Witem_dnn_layer/item_dnn_layer/item_dnn_2/dense/kernel/part_0/Initializer/random_uniformAdd[item_dnn_layer/item_dnn_layer/item_dnn_2/dense/kernel/part_0/Initializer/random_uniform/mul[item_dnn_layer/item_dnn_layer/item_dnn_2/dense/kernel/part_0/Initializer/random_uniform/min*
_output_shapes
:	@*O
_classE
CAloc:@item_dnn_layer/item_dnn_layer/item_dnn_2/dense/kernel/part_0*
T0
ß
<item_dnn_layer/item_dnn_layer/item_dnn_2/dense/kernel/part_0
VariableV2*
shape:	@*
_output_shapes
:	@*
dtype0*O
_classE
CAloc:@item_dnn_layer/item_dnn_layer/item_dnn_2/dense/kernel/part_0
ß
Citem_dnn_layer/item_dnn_layer/item_dnn_2/dense/kernel/part_0/AssignAssign<item_dnn_layer/item_dnn_layer/item_dnn_2/dense/kernel/part_0Witem_dnn_layer/item_dnn_layer/item_dnn_2/dense/kernel/part_0/Initializer/random_uniform*O
_classE
CAloc:@item_dnn_layer/item_dnn_layer/item_dnn_2/dense/kernel/part_0*
T0*
_output_shapes
:	@

Aitem_dnn_layer/item_dnn_layer/item_dnn_2/dense/kernel/part_0/readIdentity<item_dnn_layer/item_dnn_layer/item_dnn_2/dense/kernel/part_0*
T0*O
_classE
CAloc:@item_dnn_layer/item_dnn_layer/item_dnn_2/dense/kernel/part_0*
_output_shapes
:	@
č
Litem_dnn_layer/item_dnn_layer/item_dnn_2/dense/bias/part_0/Initializer/zerosConst*
dtype0*M
_classC
A?loc:@item_dnn_layer/item_dnn_layer/item_dnn_2/dense/bias/part_0*
valueB@*    *
_output_shapes
:@
Ń
:item_dnn_layer/item_dnn_layer/item_dnn_2/dense/bias/part_0
VariableV2*
_output_shapes
:@*M
_classC
A?loc:@item_dnn_layer/item_dnn_layer/item_dnn_2/dense/bias/part_0*
shape:@*
dtype0
É
Aitem_dnn_layer/item_dnn_layer/item_dnn_2/dense/bias/part_0/AssignAssign:item_dnn_layer/item_dnn_layer/item_dnn_2/dense/bias/part_0Litem_dnn_layer/item_dnn_layer/item_dnn_2/dense/bias/part_0/Initializer/zeros*
_output_shapes
:@*
T0*M
_classC
A?loc:@item_dnn_layer/item_dnn_layer/item_dnn_2/dense/bias/part_0
ű
?item_dnn_layer/item_dnn_layer/item_dnn_2/dense/bias/part_0/readIdentity:item_dnn_layer/item_dnn_layer/item_dnn_2/dense/bias/part_0*M
_classC
A?loc:@item_dnn_layer/item_dnn_layer/item_dnn_2/dense/bias/part_0*
_output_shapes
:@*
T0
Ž
5item_dnn_layer/item_dnn_layer/item_dnn_2/dense/kernelIdentityAitem_dnn_layer/item_dnn_layer/item_dnn_2/dense/kernel/part_0/read*
T0*
_output_shapes
:	@
×
5item_dnn_layer/item_dnn_layer/item_dnn_2/dense/MatMulMatMul-item_dnn_layer/item_dnn_layer/item_dnn_1/Relu5item_dnn_layer/item_dnn_layer/item_dnn_2/dense/kernel*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@*
T0
Ľ
3item_dnn_layer/item_dnn_layer/item_dnn_2/dense/biasIdentity?item_dnn_layer/item_dnn_layer/item_dnn_2/dense/bias/part_0/read*
T0*
_output_shapes
:@
ß
6item_dnn_layer/item_dnn_layer/item_dnn_2/dense/BiasAddBiasAdd5item_dnn_layer/item_dnn_layer/item_dnn_2/dense/MatMul3item_dnn_layer/item_dnn_layer/item_dnn_2/dense/bias*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@*
T0

Zitem_dnn_layer/item_dnn_layer/item_dnn_2/batch_normalization/gamma/part_0/Initializer/onesConst*\
_classR
PNloc:@item_dnn_layer/item_dnn_layer/item_dnn_2/batch_normalization/gamma/part_0*
dtype0*
_output_shapes
:@*
valueB@*  ?
ď
Iitem_dnn_layer/item_dnn_layer/item_dnn_2/batch_normalization/gamma/part_0
VariableV2*
dtype0*
_output_shapes
:@*\
_classR
PNloc:@item_dnn_layer/item_dnn_layer/item_dnn_2/batch_normalization/gamma/part_0*
shape:@

Pitem_dnn_layer/item_dnn_layer/item_dnn_2/batch_normalization/gamma/part_0/AssignAssignIitem_dnn_layer/item_dnn_layer/item_dnn_2/batch_normalization/gamma/part_0Zitem_dnn_layer/item_dnn_layer/item_dnn_2/batch_normalization/gamma/part_0/Initializer/ones*
_output_shapes
:@*
T0*\
_classR
PNloc:@item_dnn_layer/item_dnn_layer/item_dnn_2/batch_normalization/gamma/part_0
¨
Nitem_dnn_layer/item_dnn_layer/item_dnn_2/batch_normalization/gamma/part_0/readIdentityIitem_dnn_layer/item_dnn_layer/item_dnn_2/batch_normalization/gamma/part_0*\
_classR
PNloc:@item_dnn_layer/item_dnn_layer/item_dnn_2/batch_normalization/gamma/part_0*
_output_shapes
:@*
T0

Zitem_dnn_layer/item_dnn_layer/item_dnn_2/batch_normalization/beta/part_0/Initializer/zerosConst*
dtype0*[
_classQ
OMloc:@item_dnn_layer/item_dnn_layer/item_dnn_2/batch_normalization/beta/part_0*
_output_shapes
:@*
valueB@*    
í
Hitem_dnn_layer/item_dnn_layer/item_dnn_2/batch_normalization/beta/part_0
VariableV2*
_output_shapes
:@*
dtype0*
shape:@*[
_classQ
OMloc:@item_dnn_layer/item_dnn_layer/item_dnn_2/batch_normalization/beta/part_0

Oitem_dnn_layer/item_dnn_layer/item_dnn_2/batch_normalization/beta/part_0/AssignAssignHitem_dnn_layer/item_dnn_layer/item_dnn_2/batch_normalization/beta/part_0Zitem_dnn_layer/item_dnn_layer/item_dnn_2/batch_normalization/beta/part_0/Initializer/zeros*
T0*
_output_shapes
:@*[
_classQ
OMloc:@item_dnn_layer/item_dnn_layer/item_dnn_2/batch_normalization/beta/part_0
Ľ
Mitem_dnn_layer/item_dnn_layer/item_dnn_2/batch_normalization/beta/part_0/readIdentityHitem_dnn_layer/item_dnn_layer/item_dnn_2/batch_normalization/beta/part_0*[
_classQ
OMloc:@item_dnn_layer/item_dnn_layer/item_dnn_2/batch_normalization/beta/part_0*
T0*
_output_shapes
:@

Zitem_dnn_layer/item_dnn_layer/item_dnn_2/batch_normalization/moving_mean/Initializer/zerosConst*[
_classQ
OMloc:@item_dnn_layer/item_dnn_layer/item_dnn_2/batch_normalization/moving_mean*
valueB@*    *
_output_shapes
:@*
dtype0
í
Hitem_dnn_layer/item_dnn_layer/item_dnn_2/batch_normalization/moving_mean
VariableV2*
_output_shapes
:@*
dtype0*
shape:@*[
_classQ
OMloc:@item_dnn_layer/item_dnn_layer/item_dnn_2/batch_normalization/moving_mean

Oitem_dnn_layer/item_dnn_layer/item_dnn_2/batch_normalization/moving_mean/AssignAssignHitem_dnn_layer/item_dnn_layer/item_dnn_2/batch_normalization/moving_meanZitem_dnn_layer/item_dnn_layer/item_dnn_2/batch_normalization/moving_mean/Initializer/zeros*
T0*
_output_shapes
:@*[
_classQ
OMloc:@item_dnn_layer/item_dnn_layer/item_dnn_2/batch_normalization/moving_mean
Ľ
Mitem_dnn_layer/item_dnn_layer/item_dnn_2/batch_normalization/moving_mean/readIdentityHitem_dnn_layer/item_dnn_layer/item_dnn_2/batch_normalization/moving_mean*[
_classQ
OMloc:@item_dnn_layer/item_dnn_layer/item_dnn_2/batch_normalization/moving_mean*
_output_shapes
:@*
T0

]item_dnn_layer/item_dnn_layer/item_dnn_2/batch_normalization/moving_variance/Initializer/onesConst*_
_classU
SQloc:@item_dnn_layer/item_dnn_layer/item_dnn_2/batch_normalization/moving_variance*
valueB@*  ?*
_output_shapes
:@*
dtype0
ő
Litem_dnn_layer/item_dnn_layer/item_dnn_2/batch_normalization/moving_variance
VariableV2*
dtype0*
_output_shapes
:@*
shape:@*_
_classU
SQloc:@item_dnn_layer/item_dnn_layer/item_dnn_2/batch_normalization/moving_variance

Sitem_dnn_layer/item_dnn_layer/item_dnn_2/batch_normalization/moving_variance/AssignAssignLitem_dnn_layer/item_dnn_layer/item_dnn_2/batch_normalization/moving_variance]item_dnn_layer/item_dnn_layer/item_dnn_2/batch_normalization/moving_variance/Initializer/ones*_
_classU
SQloc:@item_dnn_layer/item_dnn_layer/item_dnn_2/batch_normalization/moving_variance*
_output_shapes
:@*
T0
ą
Qitem_dnn_layer/item_dnn_layer/item_dnn_2/batch_normalization/moving_variance/readIdentityLitem_dnn_layer/item_dnn_layer/item_dnn_2/batch_normalization/moving_variance*
_output_shapes
:@*
T0*_
_classU
SQloc:@item_dnn_layer/item_dnn_layer/item_dnn_2/batch_normalization/moving_variance
Ľ
[item_dnn_layer/item_dnn_layer/item_dnn_2/batch_normalization/moments/mean/reduction_indicesConst*
valueB: *
_output_shapes
:*
dtype0
 
Iitem_dnn_layer/item_dnn_layer/item_dnn_2/batch_normalization/moments/meanMean6item_dnn_layer/item_dnn_layer/item_dnn_2/dense/BiasAdd[item_dnn_layer/item_dnn_layer/item_dnn_2/batch_normalization/moments/mean/reduction_indices*
_output_shapes

:@*
T0*
	keep_dims(
Ő
Qitem_dnn_layer/item_dnn_layer/item_dnn_2/batch_normalization/moments/StopGradientStopGradientIitem_dnn_layer/item_dnn_layer/item_dnn_2/batch_normalization/moments/mean*
_output_shapes

:@*
T0
¨
Vitem_dnn_layer/item_dnn_layer/item_dnn_2/batch_normalization/moments/SquaredDifferenceSquaredDifference6item_dnn_layer/item_dnn_layer/item_dnn_2/dense/BiasAddQitem_dnn_layer/item_dnn_layer/item_dnn_2/batch_normalization/moments/StopGradient*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@
Š
_item_dnn_layer/item_dnn_layer/item_dnn_2/batch_normalization/moments/variance/reduction_indicesConst*
valueB: *
_output_shapes
:*
dtype0
Č
Mitem_dnn_layer/item_dnn_layer/item_dnn_2/batch_normalization/moments/varianceMeanVitem_dnn_layer/item_dnn_layer/item_dnn_2/batch_normalization/moments/SquaredDifference_item_dnn_layer/item_dnn_layer/item_dnn_2/batch_normalization/moments/variance/reduction_indices*
_output_shapes

:@*
	keep_dims(*
T0
Ţ
Litem_dnn_layer/item_dnn_layer/item_dnn_2/batch_normalization/moments/SqueezeSqueezeIitem_dnn_layer/item_dnn_layer/item_dnn_2/batch_normalization/moments/mean*
T0*
_output_shapes
:@*
squeeze_dims
 
ä
Nitem_dnn_layer/item_dnn_layer/item_dnn_2/batch_normalization/moments/Squeeze_1SqueezeMitem_dnn_layer/item_dnn_layer/item_dnn_2/batch_normalization/moments/variance*
T0*
squeeze_dims
 *
_output_shapes
:@
ô
Ritem_dnn_layer/item_dnn_layer/item_dnn_2/batch_normalization/AssignMovingAvg/decayConst*[
_classQ
OMloc:@item_dnn_layer/item_dnn_layer/item_dnn_2/batch_normalization/moving_mean*
dtype0*
_output_shapes
: *
valueB
 *
×#<
ö
Pitem_dnn_layer/item_dnn_layer/item_dnn_2/batch_normalization/AssignMovingAvg/subSubMitem_dnn_layer/item_dnn_layer/item_dnn_2/batch_normalization/moving_mean/readLitem_dnn_layer/item_dnn_layer/item_dnn_2/batch_normalization/moments/Squeeze*
_output_shapes
:@*[
_classQ
OMloc:@item_dnn_layer/item_dnn_layer/item_dnn_2/batch_normalization/moving_mean*
T0
˙
Pitem_dnn_layer/item_dnn_layer/item_dnn_2/batch_normalization/AssignMovingAvg/mulMulPitem_dnn_layer/item_dnn_layer/item_dnn_2/batch_normalization/AssignMovingAvg/subRitem_dnn_layer/item_dnn_layer/item_dnn_2/batch_normalization/AssignMovingAvg/decay*[
_classQ
OMloc:@item_dnn_layer/item_dnn_layer/item_dnn_2/batch_normalization/moving_mean*
T0*
_output_shapes
:@
÷
Litem_dnn_layer/item_dnn_layer/item_dnn_2/batch_normalization/AssignMovingAvg	AssignSubHitem_dnn_layer/item_dnn_layer/item_dnn_2/batch_normalization/moving_meanPitem_dnn_layer/item_dnn_layer/item_dnn_2/batch_normalization/AssignMovingAvg/mul*
T0*
_output_shapes
:@*[
_classQ
OMloc:@item_dnn_layer/item_dnn_layer/item_dnn_2/batch_normalization/moving_mean
ú
Titem_dnn_layer/item_dnn_layer/item_dnn_2/batch_normalization/AssignMovingAvg_1/decayConst*_
_classU
SQloc:@item_dnn_layer/item_dnn_layer/item_dnn_2/batch_normalization/moving_variance*
dtype0*
valueB
 *
×#<*
_output_shapes
: 

Ritem_dnn_layer/item_dnn_layer/item_dnn_2/batch_normalization/AssignMovingAvg_1/subSubQitem_dnn_layer/item_dnn_layer/item_dnn_2/batch_normalization/moving_variance/readNitem_dnn_layer/item_dnn_layer/item_dnn_2/batch_normalization/moments/Squeeze_1*
_output_shapes
:@*
T0*_
_classU
SQloc:@item_dnn_layer/item_dnn_layer/item_dnn_2/batch_normalization/moving_variance

Ritem_dnn_layer/item_dnn_layer/item_dnn_2/batch_normalization/AssignMovingAvg_1/mulMulRitem_dnn_layer/item_dnn_layer/item_dnn_2/batch_normalization/AssignMovingAvg_1/subTitem_dnn_layer/item_dnn_layer/item_dnn_2/batch_normalization/AssignMovingAvg_1/decay*
_output_shapes
:@*_
_classU
SQloc:@item_dnn_layer/item_dnn_layer/item_dnn_2/batch_normalization/moving_variance*
T0

Nitem_dnn_layer/item_dnn_layer/item_dnn_2/batch_normalization/AssignMovingAvg_1	AssignSubLitem_dnn_layer/item_dnn_layer/item_dnn_2/batch_normalization/moving_varianceRitem_dnn_layer/item_dnn_layer/item_dnn_2/batch_normalization/AssignMovingAvg_1/mul*
T0*
_output_shapes
:@*_
_classU
SQloc:@item_dnn_layer/item_dnn_layer/item_dnn_2/batch_normalization/moving_variance
Á
Aitem_dnn_layer/item_dnn_layer/item_dnn_2/batch_normalization/betaIdentityMitem_dnn_layer/item_dnn_layer/item_dnn_2/batch_normalization/beta/part_0/read*
T0*
_output_shapes
:@
Ă
Bitem_dnn_layer/item_dnn_layer/item_dnn_2/batch_normalization/gammaIdentityNitem_dnn_layer/item_dnn_layer/item_dnn_2/batch_normalization/gamma/part_0/read*
_output_shapes
:@*
T0

Litem_dnn_layer/item_dnn_layer/item_dnn_2/batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
valueB
 *o:*
dtype0

Jitem_dnn_layer/item_dnn_layer/item_dnn_2/batch_normalization/batchnorm/addAddV2Nitem_dnn_layer/item_dnn_layer/item_dnn_2/batch_normalization/moments/Squeeze_1Litem_dnn_layer/item_dnn_layer/item_dnn_2/batch_normalization/batchnorm/add/y*
T0*
_output_shapes
:@
Ć
Litem_dnn_layer/item_dnn_layer/item_dnn_2/batch_normalization/batchnorm/RsqrtRsqrtJitem_dnn_layer/item_dnn_layer/item_dnn_2/batch_normalization/batchnorm/add*
T0*
_output_shapes
:@

Jitem_dnn_layer/item_dnn_layer/item_dnn_2/batch_normalization/batchnorm/mulMulLitem_dnn_layer/item_dnn_layer/item_dnn_2/batch_normalization/batchnorm/RsqrtBitem_dnn_layer/item_dnn_layer/item_dnn_2/batch_normalization/gamma*
_output_shapes
:@*
T0

Litem_dnn_layer/item_dnn_layer/item_dnn_2/batch_normalization/batchnorm/mul_1Mul6item_dnn_layer/item_dnn_layer/item_dnn_2/dense/BiasAddJitem_dnn_layer/item_dnn_layer/item_dnn_2/batch_normalization/batchnorm/mul*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@

Litem_dnn_layer/item_dnn_layer/item_dnn_2/batch_normalization/batchnorm/mul_2MulLitem_dnn_layer/item_dnn_layer/item_dnn_2/batch_normalization/moments/SqueezeJitem_dnn_layer/item_dnn_layer/item_dnn_2/batch_normalization/batchnorm/mul*
T0*
_output_shapes
:@

Jitem_dnn_layer/item_dnn_layer/item_dnn_2/batch_normalization/batchnorm/subSubAitem_dnn_layer/item_dnn_layer/item_dnn_2/batch_normalization/betaLitem_dnn_layer/item_dnn_layer/item_dnn_2/batch_normalization/batchnorm/mul_2*
_output_shapes
:@*
T0
Ą
Litem_dnn_layer/item_dnn_layer/item_dnn_2/batch_normalization/batchnorm/add_1AddV2Litem_dnn_layer/item_dnn_layer/item_dnn_2/batch_normalization/batchnorm/mul_1Jitem_dnn_layer/item_dnn_layer/item_dnn_2/batch_normalization/batchnorm/sub*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@*
T0
ľ
-item_dnn_layer/item_dnn_layer/item_dnn_2/ReluReluLitem_dnn_layer/item_dnn_layer/item_dnn_2/batch_normalization/batchnorm/add_1*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@
Ł
;item_dnn_layer/item_dnn_layer/item_dnn_2/zero_fraction/SizeSize-item_dnn_layer/item_dnn_layer/item_dnn_2/Relu*
T0*
out_type0	*
_output_shapes
: 

Bitem_dnn_layer/item_dnn_layer/item_dnn_2/zero_fraction/LessEqual/yConst*
_output_shapes
: *
valueB	 R˙˙˙˙*
dtype0	
ď
@item_dnn_layer/item_dnn_layer/item_dnn_2/zero_fraction/LessEqual	LessEqual;item_dnn_layer/item_dnn_layer/item_dnn_2/zero_fraction/SizeBitem_dnn_layer/item_dnn_layer/item_dnn_2/zero_fraction/LessEqual/y*
_output_shapes
: *
T0	
ó
Bitem_dnn_layer/item_dnn_layer/item_dnn_2/zero_fraction/cond/SwitchSwitch@item_dnn_layer/item_dnn_layer/item_dnn_2/zero_fraction/LessEqual@item_dnn_layer/item_dnn_layer/item_dnn_2/zero_fraction/LessEqual*
_output_shapes
: : *
T0

ˇ
Ditem_dnn_layer/item_dnn_layer/item_dnn_2/zero_fraction/cond/switch_tIdentityDitem_dnn_layer/item_dnn_layer/item_dnn_2/zero_fraction/cond/Switch:1*
T0
*
_output_shapes
: 
ľ
Ditem_dnn_layer/item_dnn_layer/item_dnn_2/zero_fraction/cond/switch_fIdentityBitem_dnn_layer/item_dnn_layer/item_dnn_2/zero_fraction/cond/Switch*
_output_shapes
: *
T0

˛
Citem_dnn_layer/item_dnn_layer/item_dnn_2/zero_fraction/cond/pred_idIdentity@item_dnn_layer/item_dnn_layer/item_dnn_2/zero_fraction/LessEqual*
T0
*
_output_shapes
: 
Ű
Oitem_dnn_layer/item_dnn_layer/item_dnn_2/zero_fraction/cond/count_nonzero/zerosConstE^item_dnn_layer/item_dnn_layer/item_dnn_2/zero_fraction/cond/switch_t*
valueB
 *    *
_output_shapes
: *
dtype0
ž
Ritem_dnn_layer/item_dnn_layer/item_dnn_2/zero_fraction/cond/count_nonzero/NotEqualNotEqual[item_dnn_layer/item_dnn_layer/item_dnn_2/zero_fraction/cond/count_nonzero/NotEqual/Switch:1Oitem_dnn_layer/item_dnn_layer/item_dnn_2/zero_fraction/cond/count_nonzero/zeros*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@
Ţ
Yitem_dnn_layer/item_dnn_layer/item_dnn_2/zero_fraction/cond/count_nonzero/NotEqual/SwitchSwitch-item_dnn_layer/item_dnn_layer/item_dnn_2/ReluCitem_dnn_layer/item_dnn_layer/item_dnn_2/zero_fraction/cond/pred_id*
T0*:
_output_shapes(
&:˙˙˙˙˙˙˙˙˙@:˙˙˙˙˙˙˙˙˙@*@
_class6
42loc:@item_dnn_layer/item_dnn_layer/item_dnn_2/Relu
ë
Nitem_dnn_layer/item_dnn_layer/item_dnn_2/zero_fraction/cond/count_nonzero/CastCastRitem_dnn_layer/item_dnn_layer/item_dnn_2/zero_fraction/cond/count_nonzero/NotEqual*

SrcT0
*

DstT0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@
ç
Oitem_dnn_layer/item_dnn_layer/item_dnn_2/zero_fraction/cond/count_nonzero/ConstConstE^item_dnn_layer/item_dnn_layer/item_dnn_2/zero_fraction/cond/switch_t*
valueB"       *
dtype0*
_output_shapes
:
 
Witem_dnn_layer/item_dnn_layer/item_dnn_2/zero_fraction/cond/count_nonzero/nonzero_countSumNitem_dnn_layer/item_dnn_layer/item_dnn_2/zero_fraction/cond/count_nonzero/CastOitem_dnn_layer/item_dnn_layer/item_dnn_2/zero_fraction/cond/count_nonzero/Const*
T0*
_output_shapes
: 
Ń
@item_dnn_layer/item_dnn_layer/item_dnn_2/zero_fraction/cond/CastCastWitem_dnn_layer/item_dnn_layer/item_dnn_2/zero_fraction/cond/count_nonzero/nonzero_count*

DstT0	*

SrcT0*
_output_shapes
: 
Ý
Qitem_dnn_layer/item_dnn_layer/item_dnn_2/zero_fraction/cond/count_nonzero_1/zerosConstE^item_dnn_layer/item_dnn_layer/item_dnn_2/zero_fraction/cond/switch_f*
_output_shapes
: *
dtype0*
valueB
 *    
Â
Titem_dnn_layer/item_dnn_layer/item_dnn_2/zero_fraction/cond/count_nonzero_1/NotEqualNotEqual[item_dnn_layer/item_dnn_layer/item_dnn_2/zero_fraction/cond/count_nonzero_1/NotEqual/SwitchQitem_dnn_layer/item_dnn_layer/item_dnn_2/zero_fraction/cond/count_nonzero_1/zeros*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@*
T0
ŕ
[item_dnn_layer/item_dnn_layer/item_dnn_2/zero_fraction/cond/count_nonzero_1/NotEqual/SwitchSwitch-item_dnn_layer/item_dnn_layer/item_dnn_2/ReluCitem_dnn_layer/item_dnn_layer/item_dnn_2/zero_fraction/cond/pred_id*@
_class6
42loc:@item_dnn_layer/item_dnn_layer/item_dnn_2/Relu*
T0*:
_output_shapes(
&:˙˙˙˙˙˙˙˙˙@:˙˙˙˙˙˙˙˙˙@
ď
Pitem_dnn_layer/item_dnn_layer/item_dnn_2/zero_fraction/cond/count_nonzero_1/CastCastTitem_dnn_layer/item_dnn_layer/item_dnn_2/zero_fraction/cond/count_nonzero_1/NotEqual*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@*

DstT0	*

SrcT0

é
Qitem_dnn_layer/item_dnn_layer/item_dnn_2/zero_fraction/cond/count_nonzero_1/ConstConstE^item_dnn_layer/item_dnn_layer/item_dnn_2/zero_fraction/cond/switch_f*
valueB"       *
_output_shapes
:*
dtype0
Ś
Yitem_dnn_layer/item_dnn_layer/item_dnn_2/zero_fraction/cond/count_nonzero_1/nonzero_countSumPitem_dnn_layer/item_dnn_layer/item_dnn_2/zero_fraction/cond/count_nonzero_1/CastQitem_dnn_layer/item_dnn_layer/item_dnn_2/zero_fraction/cond/count_nonzero_1/Const*
T0	*
_output_shapes
: 

Aitem_dnn_layer/item_dnn_layer/item_dnn_2/zero_fraction/cond/MergeMergeYitem_dnn_layer/item_dnn_layer/item_dnn_2/zero_fraction/cond/count_nonzero_1/nonzero_count@item_dnn_layer/item_dnn_layer/item_dnn_2/zero_fraction/cond/Cast*
_output_shapes
: : *
N*
T0	
ő
Mitem_dnn_layer/item_dnn_layer/item_dnn_2/zero_fraction/counts_to_fraction/subSub;item_dnn_layer/item_dnn_layer/item_dnn_2/zero_fraction/SizeAitem_dnn_layer/item_dnn_layer/item_dnn_2/zero_fraction/cond/Merge*
T0	*
_output_shapes
: 
Ő
Nitem_dnn_layer/item_dnn_layer/item_dnn_2/zero_fraction/counts_to_fraction/CastCastMitem_dnn_layer/item_dnn_layer/item_dnn_2/zero_fraction/counts_to_fraction/sub*
_output_shapes
: *

DstT0*

SrcT0	
Ĺ
Pitem_dnn_layer/item_dnn_layer/item_dnn_2/zero_fraction/counts_to_fraction/Cast_1Cast;item_dnn_layer/item_dnn_layer/item_dnn_2/zero_fraction/Size*
_output_shapes
: *

DstT0*

SrcT0	

Qitem_dnn_layer/item_dnn_layer/item_dnn_2/zero_fraction/counts_to_fraction/truedivRealDivNitem_dnn_layer/item_dnn_layer/item_dnn_2/zero_fraction/counts_to_fraction/CastPitem_dnn_layer/item_dnn_layer/item_dnn_2/zero_fraction/counts_to_fraction/Cast_1*
_output_shapes
: *
T0
ż
?item_dnn_layer/item_dnn_layer/item_dnn_2/zero_fraction/fractionIdentityQitem_dnn_layer/item_dnn_layer/item_dnn_2/zero_fraction/counts_to_fraction/truediv*
T0*
_output_shapes
: 

nitem_dnn_layer/item_dnn_layer/item_dnn_2/item_dnn_layer/item_dnn_layer/item_dnn_2/fraction_of_zero_values/tagsConst*z
valueqBo Biitem_dnn_layer/item_dnn_layer/item_dnn_2/item_dnn_layer/item_dnn_layer/item_dnn_2/fraction_of_zero_values*
dtype0*
_output_shapes
: 
Ě
iitem_dnn_layer/item_dnn_layer/item_dnn_2/item_dnn_layer/item_dnn_layer/item_dnn_2/fraction_of_zero_valuesScalarSummarynitem_dnn_layer/item_dnn_layer/item_dnn_2/item_dnn_layer/item_dnn_layer/item_dnn_2/fraction_of_zero_values/tags?item_dnn_layer/item_dnn_layer/item_dnn_2/zero_fraction/fraction*
_output_shapes
: *
T0
ý
`item_dnn_layer/item_dnn_layer/item_dnn_2/item_dnn_layer/item_dnn_layer/item_dnn_2/activation/tagConst*
dtype0*m
valuedBb B\item_dnn_layer/item_dnn_layer/item_dnn_2/item_dnn_layer/item_dnn_layer/item_dnn_2/activation*
_output_shapes
: 

\item_dnn_layer/item_dnn_layer/item_dnn_2/item_dnn_layer/item_dnn_layer/item_dnn_2/activationHistogramSummary`item_dnn_layer/item_dnn_layer/item_dnn_2/item_dnn_layer/item_dnn_layer/item_dnn_2/activation/tag-item_dnn_layer/item_dnn_layer/item_dnn_2/Relu*
_output_shapes
: 
˙
]item_dnn_layer/item_dnn_layer/item_dnn_3/dense/kernel/part_0/Initializer/random_uniform/shapeConst*
valueB"@       *O
_classE
CAloc:@item_dnn_layer/item_dnn_layer/item_dnn_3/dense/kernel/part_0*
dtype0*
_output_shapes
:
ń
[item_dnn_layer/item_dnn_layer/item_dnn_3/dense/kernel/part_0/Initializer/random_uniform/minConst*
dtype0*
valueB
 *  ž*O
_classE
CAloc:@item_dnn_layer/item_dnn_layer/item_dnn_3/dense/kernel/part_0*
_output_shapes
: 
ń
[item_dnn_layer/item_dnn_layer/item_dnn_3/dense/kernel/part_0/Initializer/random_uniform/maxConst*
_output_shapes
: *
valueB
 *  >*
dtype0*O
_classE
CAloc:@item_dnn_layer/item_dnn_layer/item_dnn_3/dense/kernel/part_0
Ü
eitem_dnn_layer/item_dnn_layer/item_dnn_3/dense/kernel/part_0/Initializer/random_uniform/RandomUniformRandomUniform]item_dnn_layer/item_dnn_layer/item_dnn_3/dense/kernel/part_0/Initializer/random_uniform/shape*O
_classE
CAloc:@item_dnn_layer/item_dnn_layer/item_dnn_3/dense/kernel/part_0*
dtype0*
_output_shapes

:@ *
T0

[item_dnn_layer/item_dnn_layer/item_dnn_3/dense/kernel/part_0/Initializer/random_uniform/subSub[item_dnn_layer/item_dnn_layer/item_dnn_3/dense/kernel/part_0/Initializer/random_uniform/max[item_dnn_layer/item_dnn_layer/item_dnn_3/dense/kernel/part_0/Initializer/random_uniform/min*
_output_shapes
: *
T0*O
_classE
CAloc:@item_dnn_layer/item_dnn_layer/item_dnn_3/dense/kernel/part_0
 
[item_dnn_layer/item_dnn_layer/item_dnn_3/dense/kernel/part_0/Initializer/random_uniform/mulMuleitem_dnn_layer/item_dnn_layer/item_dnn_3/dense/kernel/part_0/Initializer/random_uniform/RandomUniform[item_dnn_layer/item_dnn_layer/item_dnn_3/dense/kernel/part_0/Initializer/random_uniform/sub*
T0*
_output_shapes

:@ *O
_classE
CAloc:@item_dnn_layer/item_dnn_layer/item_dnn_3/dense/kernel/part_0

Witem_dnn_layer/item_dnn_layer/item_dnn_3/dense/kernel/part_0/Initializer/random_uniformAdd[item_dnn_layer/item_dnn_layer/item_dnn_3/dense/kernel/part_0/Initializer/random_uniform/mul[item_dnn_layer/item_dnn_layer/item_dnn_3/dense/kernel/part_0/Initializer/random_uniform/min*
_output_shapes

:@ *
T0*O
_classE
CAloc:@item_dnn_layer/item_dnn_layer/item_dnn_3/dense/kernel/part_0
Ý
<item_dnn_layer/item_dnn_layer/item_dnn_3/dense/kernel/part_0
VariableV2*O
_classE
CAloc:@item_dnn_layer/item_dnn_layer/item_dnn_3/dense/kernel/part_0*
dtype0*
shape
:@ *
_output_shapes

:@ 
Ţ
Citem_dnn_layer/item_dnn_layer/item_dnn_3/dense/kernel/part_0/AssignAssign<item_dnn_layer/item_dnn_layer/item_dnn_3/dense/kernel/part_0Witem_dnn_layer/item_dnn_layer/item_dnn_3/dense/kernel/part_0/Initializer/random_uniform*O
_classE
CAloc:@item_dnn_layer/item_dnn_layer/item_dnn_3/dense/kernel/part_0*
T0*
_output_shapes

:@ 

Aitem_dnn_layer/item_dnn_layer/item_dnn_3/dense/kernel/part_0/readIdentity<item_dnn_layer/item_dnn_layer/item_dnn_3/dense/kernel/part_0*
T0*
_output_shapes

:@ *O
_classE
CAloc:@item_dnn_layer/item_dnn_layer/item_dnn_3/dense/kernel/part_0
č
Litem_dnn_layer/item_dnn_layer/item_dnn_3/dense/bias/part_0/Initializer/zerosConst*
valueB *    *
dtype0*
_output_shapes
: *M
_classC
A?loc:@item_dnn_layer/item_dnn_layer/item_dnn_3/dense/bias/part_0
Ń
:item_dnn_layer/item_dnn_layer/item_dnn_3/dense/bias/part_0
VariableV2*
shape: *M
_classC
A?loc:@item_dnn_layer/item_dnn_layer/item_dnn_3/dense/bias/part_0*
_output_shapes
: *
dtype0
É
Aitem_dnn_layer/item_dnn_layer/item_dnn_3/dense/bias/part_0/AssignAssign:item_dnn_layer/item_dnn_layer/item_dnn_3/dense/bias/part_0Litem_dnn_layer/item_dnn_layer/item_dnn_3/dense/bias/part_0/Initializer/zeros*
T0*
_output_shapes
: *M
_classC
A?loc:@item_dnn_layer/item_dnn_layer/item_dnn_3/dense/bias/part_0
ű
?item_dnn_layer/item_dnn_layer/item_dnn_3/dense/bias/part_0/readIdentity:item_dnn_layer/item_dnn_layer/item_dnn_3/dense/bias/part_0*
_output_shapes
: *M
_classC
A?loc:@item_dnn_layer/item_dnn_layer/item_dnn_3/dense/bias/part_0*
T0
­
5item_dnn_layer/item_dnn_layer/item_dnn_3/dense/kernelIdentityAitem_dnn_layer/item_dnn_layer/item_dnn_3/dense/kernel/part_0/read*
T0*
_output_shapes

:@ 
×
5item_dnn_layer/item_dnn_layer/item_dnn_3/dense/MatMulMatMul-item_dnn_layer/item_dnn_layer/item_dnn_2/Relu5item_dnn_layer/item_dnn_layer/item_dnn_3/dense/kernel*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
Ľ
3item_dnn_layer/item_dnn_layer/item_dnn_3/dense/biasIdentity?item_dnn_layer/item_dnn_layer/item_dnn_3/dense/bias/part_0/read*
T0*
_output_shapes
: 
ß
6item_dnn_layer/item_dnn_layer/item_dnn_3/dense/BiasAddBiasAdd5item_dnn_layer/item_dnn_layer/item_dnn_3/dense/MatMul3item_dnn_layer/item_dnn_layer/item_dnn_3/dense/bias*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ *
T0
Ź
;item_dnn_layer/item_dnn_layer/item_dnn_3/zero_fraction/SizeSize6item_dnn_layer/item_dnn_layer/item_dnn_3/dense/BiasAdd*
_output_shapes
: *
T0*
out_type0	

Bitem_dnn_layer/item_dnn_layer/item_dnn_3/zero_fraction/LessEqual/yConst*
dtype0	*
_output_shapes
: *
valueB	 R˙˙˙˙
ď
@item_dnn_layer/item_dnn_layer/item_dnn_3/zero_fraction/LessEqual	LessEqual;item_dnn_layer/item_dnn_layer/item_dnn_3/zero_fraction/SizeBitem_dnn_layer/item_dnn_layer/item_dnn_3/zero_fraction/LessEqual/y*
T0	*
_output_shapes
: 
ó
Bitem_dnn_layer/item_dnn_layer/item_dnn_3/zero_fraction/cond/SwitchSwitch@item_dnn_layer/item_dnn_layer/item_dnn_3/zero_fraction/LessEqual@item_dnn_layer/item_dnn_layer/item_dnn_3/zero_fraction/LessEqual*
_output_shapes
: : *
T0

ˇ
Ditem_dnn_layer/item_dnn_layer/item_dnn_3/zero_fraction/cond/switch_tIdentityDitem_dnn_layer/item_dnn_layer/item_dnn_3/zero_fraction/cond/Switch:1*
T0
*
_output_shapes
: 
ľ
Ditem_dnn_layer/item_dnn_layer/item_dnn_3/zero_fraction/cond/switch_fIdentityBitem_dnn_layer/item_dnn_layer/item_dnn_3/zero_fraction/cond/Switch*
_output_shapes
: *
T0

˛
Citem_dnn_layer/item_dnn_layer/item_dnn_3/zero_fraction/cond/pred_idIdentity@item_dnn_layer/item_dnn_layer/item_dnn_3/zero_fraction/LessEqual*
_output_shapes
: *
T0

Ű
Oitem_dnn_layer/item_dnn_layer/item_dnn_3/zero_fraction/cond/count_nonzero/zerosConstE^item_dnn_layer/item_dnn_layer/item_dnn_3/zero_fraction/cond/switch_t*
dtype0*
valueB
 *    *
_output_shapes
: 
ž
Ritem_dnn_layer/item_dnn_layer/item_dnn_3/zero_fraction/cond/count_nonzero/NotEqualNotEqual[item_dnn_layer/item_dnn_layer/item_dnn_3/zero_fraction/cond/count_nonzero/NotEqual/Switch:1Oitem_dnn_layer/item_dnn_layer/item_dnn_3/zero_fraction/cond/count_nonzero/zeros*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ *
T0
đ
Yitem_dnn_layer/item_dnn_layer/item_dnn_3/zero_fraction/cond/count_nonzero/NotEqual/SwitchSwitch6item_dnn_layer/item_dnn_layer/item_dnn_3/dense/BiasAddCitem_dnn_layer/item_dnn_layer/item_dnn_3/zero_fraction/cond/pred_id*I
_class?
=;loc:@item_dnn_layer/item_dnn_layer/item_dnn_3/dense/BiasAdd*
T0*:
_output_shapes(
&:˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ 
ë
Nitem_dnn_layer/item_dnn_layer/item_dnn_3/zero_fraction/cond/count_nonzero/CastCastRitem_dnn_layer/item_dnn_layer/item_dnn_3/zero_fraction/cond/count_nonzero/NotEqual*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ *

SrcT0
*

DstT0
ç
Oitem_dnn_layer/item_dnn_layer/item_dnn_3/zero_fraction/cond/count_nonzero/ConstConstE^item_dnn_layer/item_dnn_layer/item_dnn_3/zero_fraction/cond/switch_t*
_output_shapes
:*
valueB"       *
dtype0
 
Witem_dnn_layer/item_dnn_layer/item_dnn_3/zero_fraction/cond/count_nonzero/nonzero_countSumNitem_dnn_layer/item_dnn_layer/item_dnn_3/zero_fraction/cond/count_nonzero/CastOitem_dnn_layer/item_dnn_layer/item_dnn_3/zero_fraction/cond/count_nonzero/Const*
_output_shapes
: *
T0
Ń
@item_dnn_layer/item_dnn_layer/item_dnn_3/zero_fraction/cond/CastCastWitem_dnn_layer/item_dnn_layer/item_dnn_3/zero_fraction/cond/count_nonzero/nonzero_count*

DstT0	*
_output_shapes
: *

SrcT0
Ý
Qitem_dnn_layer/item_dnn_layer/item_dnn_3/zero_fraction/cond/count_nonzero_1/zerosConstE^item_dnn_layer/item_dnn_layer/item_dnn_3/zero_fraction/cond/switch_f*
_output_shapes
: *
valueB
 *    *
dtype0
Â
Titem_dnn_layer/item_dnn_layer/item_dnn_3/zero_fraction/cond/count_nonzero_1/NotEqualNotEqual[item_dnn_layer/item_dnn_layer/item_dnn_3/zero_fraction/cond/count_nonzero_1/NotEqual/SwitchQitem_dnn_layer/item_dnn_layer/item_dnn_3/zero_fraction/cond/count_nonzero_1/zeros*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
ň
[item_dnn_layer/item_dnn_layer/item_dnn_3/zero_fraction/cond/count_nonzero_1/NotEqual/SwitchSwitch6item_dnn_layer/item_dnn_layer/item_dnn_3/dense/BiasAddCitem_dnn_layer/item_dnn_layer/item_dnn_3/zero_fraction/cond/pred_id*
T0*:
_output_shapes(
&:˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ *I
_class?
=;loc:@item_dnn_layer/item_dnn_layer/item_dnn_3/dense/BiasAdd
ď
Pitem_dnn_layer/item_dnn_layer/item_dnn_3/zero_fraction/cond/count_nonzero_1/CastCastTitem_dnn_layer/item_dnn_layer/item_dnn_3/zero_fraction/cond/count_nonzero_1/NotEqual*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ *

SrcT0
*

DstT0	
é
Qitem_dnn_layer/item_dnn_layer/item_dnn_3/zero_fraction/cond/count_nonzero_1/ConstConstE^item_dnn_layer/item_dnn_layer/item_dnn_3/zero_fraction/cond/switch_f*
_output_shapes
:*
valueB"       *
dtype0
Ś
Yitem_dnn_layer/item_dnn_layer/item_dnn_3/zero_fraction/cond/count_nonzero_1/nonzero_countSumPitem_dnn_layer/item_dnn_layer/item_dnn_3/zero_fraction/cond/count_nonzero_1/CastQitem_dnn_layer/item_dnn_layer/item_dnn_3/zero_fraction/cond/count_nonzero_1/Const*
_output_shapes
: *
T0	

Aitem_dnn_layer/item_dnn_layer/item_dnn_3/zero_fraction/cond/MergeMergeYitem_dnn_layer/item_dnn_layer/item_dnn_3/zero_fraction/cond/count_nonzero_1/nonzero_count@item_dnn_layer/item_dnn_layer/item_dnn_3/zero_fraction/cond/Cast*
_output_shapes
: : *
N*
T0	
ő
Mitem_dnn_layer/item_dnn_layer/item_dnn_3/zero_fraction/counts_to_fraction/subSub;item_dnn_layer/item_dnn_layer/item_dnn_3/zero_fraction/SizeAitem_dnn_layer/item_dnn_layer/item_dnn_3/zero_fraction/cond/Merge*
T0	*
_output_shapes
: 
Ő
Nitem_dnn_layer/item_dnn_layer/item_dnn_3/zero_fraction/counts_to_fraction/CastCastMitem_dnn_layer/item_dnn_layer/item_dnn_3/zero_fraction/counts_to_fraction/sub*
_output_shapes
: *

SrcT0	*

DstT0
Ĺ
Pitem_dnn_layer/item_dnn_layer/item_dnn_3/zero_fraction/counts_to_fraction/Cast_1Cast;item_dnn_layer/item_dnn_layer/item_dnn_3/zero_fraction/Size*

SrcT0	*
_output_shapes
: *

DstT0

Qitem_dnn_layer/item_dnn_layer/item_dnn_3/zero_fraction/counts_to_fraction/truedivRealDivNitem_dnn_layer/item_dnn_layer/item_dnn_3/zero_fraction/counts_to_fraction/CastPitem_dnn_layer/item_dnn_layer/item_dnn_3/zero_fraction/counts_to_fraction/Cast_1*
_output_shapes
: *
T0
ż
?item_dnn_layer/item_dnn_layer/item_dnn_3/zero_fraction/fractionIdentityQitem_dnn_layer/item_dnn_layer/item_dnn_3/zero_fraction/counts_to_fraction/truediv*
T0*
_output_shapes
: 

nitem_dnn_layer/item_dnn_layer/item_dnn_3/item_dnn_layer/item_dnn_layer/item_dnn_3/fraction_of_zero_values/tagsConst*
dtype0*z
valueqBo Biitem_dnn_layer/item_dnn_layer/item_dnn_3/item_dnn_layer/item_dnn_layer/item_dnn_3/fraction_of_zero_values*
_output_shapes
: 
Ě
iitem_dnn_layer/item_dnn_layer/item_dnn_3/item_dnn_layer/item_dnn_layer/item_dnn_3/fraction_of_zero_valuesScalarSummarynitem_dnn_layer/item_dnn_layer/item_dnn_3/item_dnn_layer/item_dnn_layer/item_dnn_3/fraction_of_zero_values/tags?item_dnn_layer/item_dnn_layer/item_dnn_3/zero_fraction/fraction*
_output_shapes
: *
T0
ý
`item_dnn_layer/item_dnn_layer/item_dnn_3/item_dnn_layer/item_dnn_layer/item_dnn_3/activation/tagConst*
dtype0*
_output_shapes
: *m
valuedBb B\item_dnn_layer/item_dnn_layer/item_dnn_3/item_dnn_layer/item_dnn_layer/item_dnn_3/activation
˘
\item_dnn_layer/item_dnn_layer/item_dnn_3/item_dnn_layer/item_dnn_layer/item_dnn_3/activationHistogramSummary`item_dnn_layer/item_dnn_layer/item_dnn_3/item_dnn_layer/item_dnn_layer/item_dnn_3/activation/tag6item_dnn_layer/item_dnn_layer/item_dnn_3/dense/BiasAdd*
_output_shapes
: 

l2_normalize/SquareSquare6user_dnn_layer/user_dnn_layer/user_dnn_3/dense/BiasAdd*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ *
T0
d
"l2_normalize/Sum/reduction_indicesConst*
dtype0*
_output_shapes
: *
value	B :

l2_normalize/SumSuml2_normalize/Square"l2_normalize/Sum/reduction_indices*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
	keep_dims(*
T0
[
l2_normalize/Maximum/yConst*
valueB
 *Ěź+*
_output_shapes
: *
dtype0
{
l2_normalize/MaximumMaximuml2_normalize/Suml2_normalize/Maximum/y*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
c
l2_normalize/RsqrtRsqrtl2_normalize/Maximum*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

l2_normalizeMul6user_dnn_layer/user_dnn_layer/user_dnn_3/dense/BiasAddl2_normalize/Rsqrt*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ 

l2_normalize_1/SquareSquare6item_dnn_layer/item_dnn_layer/item_dnn_3/dense/BiasAdd*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
f
$l2_normalize_1/Sum/reduction_indicesConst*
dtype0*
_output_shapes
: *
value	B :

l2_normalize_1/SumSuml2_normalize_1/Square$l2_normalize_1/Sum/reduction_indices*
	keep_dims(*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
]
l2_normalize_1/Maximum/yConst*
dtype0*
valueB
 *Ěź+*
_output_shapes
: 

l2_normalize_1/MaximumMaximuml2_normalize_1/Suml2_normalize_1/Maximum/y*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
g
l2_normalize_1/RsqrtRsqrtl2_normalize_1/Maximum*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

l2_normalize_1Mul6item_dnn_layer/item_dnn_layer/item_dnn_3/dense/BiasAddl2_normalize_1/Rsqrt*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
Z
MulMull2_normalizel2_normalize_1*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ *
T0
W
Sum/reduction_indicesConst*
dtype0*
value	B :*
_output_shapes
: 
i
SumSumMulSum/reduction_indices*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
	keep_dims(

sim_w/Initializer/onesConst*
dtype0*
_class

loc:@sim_w*
valueB*  ?*
_output_shapes

:
o
sim_w
VariableV2*
dtype0*
_output_shapes

:*
shape
:*
_class

loc:@sim_w
x
sim_w/AssignAssignsim_wsim_w/Initializer/ones*
_class

loc:@sim_w*
T0*
_output_shapes

:
`

sim_w/readIdentitysim_w*
T0*
_class

loc:@sim_w*
_output_shapes

:
~
sim_b/Initializer/zerosConst*
valueB*    *
dtype0*
_class

loc:@sim_b*
_output_shapes
:
g
sim_b
VariableV2*
_class

loc:@sim_b*
_output_shapes
:*
shape:*
dtype0
u
sim_b/AssignAssignsim_bsim_b/Initializer/zeros*
_class

loc:@sim_b*
_output_shapes
:*
T0
\

sim_b/readIdentitysim_b*
_class

loc:@sim_b*
T0*
_output_shapes
:
?
AbsAbs
sim_w/read*
_output_shapes

:*
T0
L
MatMulMatMulSumAbs*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
R
addAddV2MatMul
sim_b/read*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
`
Reshape/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
T
ReshapeReshapeaddReshape/shape*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
I
SigmoidSigmoidReshape*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
E
RoundRoundSigmoid*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙

initNoOp

init_all_tablesNoOp

init_1NoOp
4

group_depsNoOp^init^init_1^init_all_tables
Y
save/filename/inputConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
n
save/filenamePlaceholderWithDefaultsave/filename/input*
_output_shapes
: *
dtype0*
shape: 
e

save/ConstPlaceholderWithDefaultsave/filename*
_output_shapes
: *
dtype0*
shape: 

save/StringJoin/inputs_1Const*
dtype0*<
value3B1 B+_temp_c78f0a9500174a50b605fb5c7141f6b3/part*
_output_shapes
: 
d
save/StringJoin
StringJoin
save/Constsave/StringJoin/inputs_1*
N*
_output_shapes
: 
Q
save/num_shardsConst*
dtype0*
value	B :*
_output_shapes
: 
k
save/ShardedFilename/shardConst"/device:CPU:0*
value	B : *
dtype0*
_output_shapes
: 

save/ShardedFilenameShardedFilenamesave/StringJoinsave/ShardedFilename/shardsave/num_shards"/device:CPU:0*
_output_shapes
: 
ľ
save/SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:;*Ů
valueĎBĚ;Bglobal_stepB=input_layer/input_layer/age_level_embedding/embedding_weightsB@input_layer/input_layer/cms_group_id_embedding/embedding_weightsB=input_layer/input_layer/cms_segid_embedding/embedding_weightsBHinput_layer/input_layer/new_user_class_level_embedding/embedding_weightsB>input_layer/input_layer/occupation_embedding/embedding_weightsB@input_layer/input_layer/pvalue_level_embedding/embedding_weightsBBinput_layer/input_layer/shopping_level_embedding/embedding_weightsB;input_layer/input_layer/user_id_embedding/embedding_weightsB@input_layer/input_layer_1/adgroup_id_embedding/embedding_weightsB;input_layer/input_layer_1/brand_embedding/embedding_weightsBAinput_layer/input_layer_1/campaign_id_embedding/embedding_weightsB=input_layer/input_layer_1/cate_id_embedding/embedding_weightsB>input_layer/input_layer_1/customer_embedding/embedding_weightsBGinput_layer/input_layer_1/final_gender_code_embedding/embedding_weightsB9input_layer/input_layer_1/pid_embedding/embedding_weightsB;input_layer/input_layer_1/price_embedding/embedding_weightsBAitem_dnn_layer/item_dnn_layer/item_dnn_0/batch_normalization/betaBBitem_dnn_layer/item_dnn_layer/item_dnn_0/batch_normalization/gammaBHitem_dnn_layer/item_dnn_layer/item_dnn_0/batch_normalization/moving_meanBLitem_dnn_layer/item_dnn_layer/item_dnn_0/batch_normalization/moving_varianceB3item_dnn_layer/item_dnn_layer/item_dnn_0/dense/biasB5item_dnn_layer/item_dnn_layer/item_dnn_0/dense/kernelBAitem_dnn_layer/item_dnn_layer/item_dnn_1/batch_normalization/betaBBitem_dnn_layer/item_dnn_layer/item_dnn_1/batch_normalization/gammaBHitem_dnn_layer/item_dnn_layer/item_dnn_1/batch_normalization/moving_meanBLitem_dnn_layer/item_dnn_layer/item_dnn_1/batch_normalization/moving_varianceB3item_dnn_layer/item_dnn_layer/item_dnn_1/dense/biasB5item_dnn_layer/item_dnn_layer/item_dnn_1/dense/kernelBAitem_dnn_layer/item_dnn_layer/item_dnn_2/batch_normalization/betaBBitem_dnn_layer/item_dnn_layer/item_dnn_2/batch_normalization/gammaBHitem_dnn_layer/item_dnn_layer/item_dnn_2/batch_normalization/moving_meanBLitem_dnn_layer/item_dnn_layer/item_dnn_2/batch_normalization/moving_varianceB3item_dnn_layer/item_dnn_layer/item_dnn_2/dense/biasB5item_dnn_layer/item_dnn_layer/item_dnn_2/dense/kernelB3item_dnn_layer/item_dnn_layer/item_dnn_3/dense/biasB5item_dnn_layer/item_dnn_layer/item_dnn_3/dense/kernelBsim_bBsim_wBAuser_dnn_layer/user_dnn_layer/user_dnn_0/batch_normalization/betaBBuser_dnn_layer/user_dnn_layer/user_dnn_0/batch_normalization/gammaBHuser_dnn_layer/user_dnn_layer/user_dnn_0/batch_normalization/moving_meanBLuser_dnn_layer/user_dnn_layer/user_dnn_0/batch_normalization/moving_varianceB3user_dnn_layer/user_dnn_layer/user_dnn_0/dense/biasB5user_dnn_layer/user_dnn_layer/user_dnn_0/dense/kernelBAuser_dnn_layer/user_dnn_layer/user_dnn_1/batch_normalization/betaBBuser_dnn_layer/user_dnn_layer/user_dnn_1/batch_normalization/gammaBHuser_dnn_layer/user_dnn_layer/user_dnn_1/batch_normalization/moving_meanBLuser_dnn_layer/user_dnn_layer/user_dnn_1/batch_normalization/moving_varianceB3user_dnn_layer/user_dnn_layer/user_dnn_1/dense/biasB5user_dnn_layer/user_dnn_layer/user_dnn_1/dense/kernelBAuser_dnn_layer/user_dnn_layer/user_dnn_2/batch_normalization/betaBBuser_dnn_layer/user_dnn_layer/user_dnn_2/batch_normalization/gammaBHuser_dnn_layer/user_dnn_layer/user_dnn_2/batch_normalization/moving_meanBLuser_dnn_layer/user_dnn_layer/user_dnn_2/batch_normalization/moving_varianceB3user_dnn_layer/user_dnn_layer/user_dnn_2/dense/biasB5user_dnn_layer/user_dnn_layer/user_dnn_2/dense/kernelB3user_dnn_layer/user_dnn_layer/user_dnn_3/dense/biasB5user_dnn_layer/user_dnn_layer/user_dnn_3/dense/kernel*
dtype0
˝
save/SaveV2/shape_and_slicesConst"/device:CPU:0*
dtype0*Ý
valueÓBĐ;B B10 16 0,10:0,16B100 16 0,100:0,16B100 16 0,100:0,16B10 16 0,10:0,16B10 16 0,10:0,16B10 16 0,10:0,16B10 16 0,10:0,16B100000 16 0,100000:0,16B100000 16 0,100000:0,16B100000 16 0,100000:0,16B100000 16 0,100000:0,16B10000 16 0,10000:0,16B100000 16 0,100000:0,16B10 16 0,10:0,16B10 16 0,10:0,16B50 16 0,50:0,16B	256 0,256B	256 0,256B B B	256 0,256B128 256 0,128:0,256B	128 0,128B	128 0,128B B B	128 0,128B256 128 0,256:0,128B64 0,64B64 0,64B B B64 0,64B128 64 0,128:0,64B32 0,32B64 32 0,64:0,32B B B	256 0,256B	256 0,256B B B	256 0,256B128 256 0,128:0,256B	128 0,128B	128 0,128B B B	128 0,128B256 128 0,256:0,128B64 0,64B64 0,64B B B64 0,64B128 64 0,128:0,64B32 0,32B64 32 0,64:0,32*
_output_shapes
:;
"
save/SaveV2SaveV2save/ShardedFilenamesave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesglobal_stepIinput_layer/input_layer/age_level_embedding/embedding_weights/part_0/readLinput_layer/input_layer/cms_group_id_embedding/embedding_weights/part_0/readIinput_layer/input_layer/cms_segid_embedding/embedding_weights/part_0/readTinput_layer/input_layer/new_user_class_level_embedding/embedding_weights/part_0/readJinput_layer/input_layer/occupation_embedding/embedding_weights/part_0/readLinput_layer/input_layer/pvalue_level_embedding/embedding_weights/part_0/readNinput_layer/input_layer/shopping_level_embedding/embedding_weights/part_0/readGinput_layer/input_layer/user_id_embedding/embedding_weights/part_0/readLinput_layer/input_layer_1/adgroup_id_embedding/embedding_weights/part_0/readGinput_layer/input_layer_1/brand_embedding/embedding_weights/part_0/readMinput_layer/input_layer_1/campaign_id_embedding/embedding_weights/part_0/readIinput_layer/input_layer_1/cate_id_embedding/embedding_weights/part_0/readJinput_layer/input_layer_1/customer_embedding/embedding_weights/part_0/readSinput_layer/input_layer_1/final_gender_code_embedding/embedding_weights/part_0/readEinput_layer/input_layer_1/pid_embedding/embedding_weights/part_0/readGinput_layer/input_layer_1/price_embedding/embedding_weights/part_0/readMitem_dnn_layer/item_dnn_layer/item_dnn_0/batch_normalization/beta/part_0/readNitem_dnn_layer/item_dnn_layer/item_dnn_0/batch_normalization/gamma/part_0/readHitem_dnn_layer/item_dnn_layer/item_dnn_0/batch_normalization/moving_meanLitem_dnn_layer/item_dnn_layer/item_dnn_0/batch_normalization/moving_variance?item_dnn_layer/item_dnn_layer/item_dnn_0/dense/bias/part_0/readAitem_dnn_layer/item_dnn_layer/item_dnn_0/dense/kernel/part_0/readMitem_dnn_layer/item_dnn_layer/item_dnn_1/batch_normalization/beta/part_0/readNitem_dnn_layer/item_dnn_layer/item_dnn_1/batch_normalization/gamma/part_0/readHitem_dnn_layer/item_dnn_layer/item_dnn_1/batch_normalization/moving_meanLitem_dnn_layer/item_dnn_layer/item_dnn_1/batch_normalization/moving_variance?item_dnn_layer/item_dnn_layer/item_dnn_1/dense/bias/part_0/readAitem_dnn_layer/item_dnn_layer/item_dnn_1/dense/kernel/part_0/readMitem_dnn_layer/item_dnn_layer/item_dnn_2/batch_normalization/beta/part_0/readNitem_dnn_layer/item_dnn_layer/item_dnn_2/batch_normalization/gamma/part_0/readHitem_dnn_layer/item_dnn_layer/item_dnn_2/batch_normalization/moving_meanLitem_dnn_layer/item_dnn_layer/item_dnn_2/batch_normalization/moving_variance?item_dnn_layer/item_dnn_layer/item_dnn_2/dense/bias/part_0/readAitem_dnn_layer/item_dnn_layer/item_dnn_2/dense/kernel/part_0/read?item_dnn_layer/item_dnn_layer/item_dnn_3/dense/bias/part_0/readAitem_dnn_layer/item_dnn_layer/item_dnn_3/dense/kernel/part_0/readsim_bsim_wMuser_dnn_layer/user_dnn_layer/user_dnn_0/batch_normalization/beta/part_0/readNuser_dnn_layer/user_dnn_layer/user_dnn_0/batch_normalization/gamma/part_0/readHuser_dnn_layer/user_dnn_layer/user_dnn_0/batch_normalization/moving_meanLuser_dnn_layer/user_dnn_layer/user_dnn_0/batch_normalization/moving_variance?user_dnn_layer/user_dnn_layer/user_dnn_0/dense/bias/part_0/readAuser_dnn_layer/user_dnn_layer/user_dnn_0/dense/kernel/part_0/readMuser_dnn_layer/user_dnn_layer/user_dnn_1/batch_normalization/beta/part_0/readNuser_dnn_layer/user_dnn_layer/user_dnn_1/batch_normalization/gamma/part_0/readHuser_dnn_layer/user_dnn_layer/user_dnn_1/batch_normalization/moving_meanLuser_dnn_layer/user_dnn_layer/user_dnn_1/batch_normalization/moving_variance?user_dnn_layer/user_dnn_layer/user_dnn_1/dense/bias/part_0/readAuser_dnn_layer/user_dnn_layer/user_dnn_1/dense/kernel/part_0/readMuser_dnn_layer/user_dnn_layer/user_dnn_2/batch_normalization/beta/part_0/readNuser_dnn_layer/user_dnn_layer/user_dnn_2/batch_normalization/gamma/part_0/readHuser_dnn_layer/user_dnn_layer/user_dnn_2/batch_normalization/moving_meanLuser_dnn_layer/user_dnn_layer/user_dnn_2/batch_normalization/moving_variance?user_dnn_layer/user_dnn_layer/user_dnn_2/dense/bias/part_0/readAuser_dnn_layer/user_dnn_layer/user_dnn_2/dense/kernel/part_0/read?user_dnn_layer/user_dnn_layer/user_dnn_3/dense/bias/part_0/readAuser_dnn_layer/user_dnn_layer/user_dnn_3/dense/kernel/part_0/read"/device:CPU:0*I
dtypes?
=2;	
 
save/control_dependencyIdentitysave/ShardedFilename^save/SaveV2"/device:CPU:0*
_output_shapes
: *'
_class
loc:@save/ShardedFilename*
T0
 
+save/MergeV2Checkpoints/checkpoint_prefixesPacksave/ShardedFilename^save/control_dependency"/device:CPU:0*
T0*
_output_shapes
:*
N
u
save/MergeV2CheckpointsMergeV2Checkpoints+save/MergeV2Checkpoints/checkpoint_prefixes
save/Const"/device:CPU:0

save/IdentityIdentity
save/Const^save/MergeV2Checkpoints^save/control_dependency"/device:CPU:0*
_output_shapes
: *
T0
¸
save/RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:;*Ů
valueĎBĚ;Bglobal_stepB=input_layer/input_layer/age_level_embedding/embedding_weightsB@input_layer/input_layer/cms_group_id_embedding/embedding_weightsB=input_layer/input_layer/cms_segid_embedding/embedding_weightsBHinput_layer/input_layer/new_user_class_level_embedding/embedding_weightsB>input_layer/input_layer/occupation_embedding/embedding_weightsB@input_layer/input_layer/pvalue_level_embedding/embedding_weightsBBinput_layer/input_layer/shopping_level_embedding/embedding_weightsB;input_layer/input_layer/user_id_embedding/embedding_weightsB@input_layer/input_layer_1/adgroup_id_embedding/embedding_weightsB;input_layer/input_layer_1/brand_embedding/embedding_weightsBAinput_layer/input_layer_1/campaign_id_embedding/embedding_weightsB=input_layer/input_layer_1/cate_id_embedding/embedding_weightsB>input_layer/input_layer_1/customer_embedding/embedding_weightsBGinput_layer/input_layer_1/final_gender_code_embedding/embedding_weightsB9input_layer/input_layer_1/pid_embedding/embedding_weightsB;input_layer/input_layer_1/price_embedding/embedding_weightsBAitem_dnn_layer/item_dnn_layer/item_dnn_0/batch_normalization/betaBBitem_dnn_layer/item_dnn_layer/item_dnn_0/batch_normalization/gammaBHitem_dnn_layer/item_dnn_layer/item_dnn_0/batch_normalization/moving_meanBLitem_dnn_layer/item_dnn_layer/item_dnn_0/batch_normalization/moving_varianceB3item_dnn_layer/item_dnn_layer/item_dnn_0/dense/biasB5item_dnn_layer/item_dnn_layer/item_dnn_0/dense/kernelBAitem_dnn_layer/item_dnn_layer/item_dnn_1/batch_normalization/betaBBitem_dnn_layer/item_dnn_layer/item_dnn_1/batch_normalization/gammaBHitem_dnn_layer/item_dnn_layer/item_dnn_1/batch_normalization/moving_meanBLitem_dnn_layer/item_dnn_layer/item_dnn_1/batch_normalization/moving_varianceB3item_dnn_layer/item_dnn_layer/item_dnn_1/dense/biasB5item_dnn_layer/item_dnn_layer/item_dnn_1/dense/kernelBAitem_dnn_layer/item_dnn_layer/item_dnn_2/batch_normalization/betaBBitem_dnn_layer/item_dnn_layer/item_dnn_2/batch_normalization/gammaBHitem_dnn_layer/item_dnn_layer/item_dnn_2/batch_normalization/moving_meanBLitem_dnn_layer/item_dnn_layer/item_dnn_2/batch_normalization/moving_varianceB3item_dnn_layer/item_dnn_layer/item_dnn_2/dense/biasB5item_dnn_layer/item_dnn_layer/item_dnn_2/dense/kernelB3item_dnn_layer/item_dnn_layer/item_dnn_3/dense/biasB5item_dnn_layer/item_dnn_layer/item_dnn_3/dense/kernelBsim_bBsim_wBAuser_dnn_layer/user_dnn_layer/user_dnn_0/batch_normalization/betaBBuser_dnn_layer/user_dnn_layer/user_dnn_0/batch_normalization/gammaBHuser_dnn_layer/user_dnn_layer/user_dnn_0/batch_normalization/moving_meanBLuser_dnn_layer/user_dnn_layer/user_dnn_0/batch_normalization/moving_varianceB3user_dnn_layer/user_dnn_layer/user_dnn_0/dense/biasB5user_dnn_layer/user_dnn_layer/user_dnn_0/dense/kernelBAuser_dnn_layer/user_dnn_layer/user_dnn_1/batch_normalization/betaBBuser_dnn_layer/user_dnn_layer/user_dnn_1/batch_normalization/gammaBHuser_dnn_layer/user_dnn_layer/user_dnn_1/batch_normalization/moving_meanBLuser_dnn_layer/user_dnn_layer/user_dnn_1/batch_normalization/moving_varianceB3user_dnn_layer/user_dnn_layer/user_dnn_1/dense/biasB5user_dnn_layer/user_dnn_layer/user_dnn_1/dense/kernelBAuser_dnn_layer/user_dnn_layer/user_dnn_2/batch_normalization/betaBBuser_dnn_layer/user_dnn_layer/user_dnn_2/batch_normalization/gammaBHuser_dnn_layer/user_dnn_layer/user_dnn_2/batch_normalization/moving_meanBLuser_dnn_layer/user_dnn_layer/user_dnn_2/batch_normalization/moving_varianceB3user_dnn_layer/user_dnn_layer/user_dnn_2/dense/biasB5user_dnn_layer/user_dnn_layer/user_dnn_2/dense/kernelB3user_dnn_layer/user_dnn_layer/user_dnn_3/dense/biasB5user_dnn_layer/user_dnn_layer/user_dnn_3/dense/kernel*
dtype0
Ŕ
save/RestoreV2/shape_and_slicesConst"/device:CPU:0*Ý
valueÓBĐ;B B10 16 0,10:0,16B100 16 0,100:0,16B100 16 0,100:0,16B10 16 0,10:0,16B10 16 0,10:0,16B10 16 0,10:0,16B10 16 0,10:0,16B100000 16 0,100000:0,16B100000 16 0,100000:0,16B100000 16 0,100000:0,16B100000 16 0,100000:0,16B10000 16 0,10000:0,16B100000 16 0,100000:0,16B10 16 0,10:0,16B10 16 0,10:0,16B50 16 0,50:0,16B	256 0,256B	256 0,256B B B	256 0,256B128 256 0,128:0,256B	128 0,128B	128 0,128B B B	128 0,128B256 128 0,256:0,128B64 0,64B64 0,64B B B64 0,64B128 64 0,128:0,64B32 0,32B64 32 0,64:0,32B B B	256 0,256B	256 0,256B B B	256 0,256B128 256 0,128:0,256B	128 0,128B	128 0,128B B B	128 0,128B256 128 0,256:0,128B64 0,64B64 0,64B B B64 0,64B128 64 0,128:0,64B32 0,32B64 32 0,64:0,32*
_output_shapes
:;*
dtype0

save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0*I
dtypes?
=2;	*Ű
_output_shapesČ
Ĺ::
:d:d:
:
:
:
:
 :
 :
 :
 :	N:
 :
:
:2::::::
::::::
:@:@:::@:	@: :@ ::::::::
::::::
:@:@:::@:	@: :@ 
s
save/AssignAssignglobal_stepsave/RestoreV2*
_class
loc:@global_step*
_output_shapes
: *
T0	
ń
save/Assign_1AssignDinput_layer/input_layer/age_level_embedding/embedding_weights/part_0save/RestoreV2:1*W
_classM
KIloc:@input_layer/input_layer/age_level_embedding/embedding_weights/part_0*
_output_shapes

:
*
T0
÷
save/Assign_2AssignGinput_layer/input_layer/cms_group_id_embedding/embedding_weights/part_0save/RestoreV2:2*
_output_shapes

:d*
T0*Z
_classP
NLloc:@input_layer/input_layer/cms_group_id_embedding/embedding_weights/part_0
ń
save/Assign_3AssignDinput_layer/input_layer/cms_segid_embedding/embedding_weights/part_0save/RestoreV2:3*
T0*W
_classM
KIloc:@input_layer/input_layer/cms_segid_embedding/embedding_weights/part_0*
_output_shapes

:d

save/Assign_4AssignOinput_layer/input_layer/new_user_class_level_embedding/embedding_weights/part_0save/RestoreV2:4*
_output_shapes

:
*b
_classX
VTloc:@input_layer/input_layer/new_user_class_level_embedding/embedding_weights/part_0*
T0
ó
save/Assign_5AssignEinput_layer/input_layer/occupation_embedding/embedding_weights/part_0save/RestoreV2:5*
T0*X
_classN
LJloc:@input_layer/input_layer/occupation_embedding/embedding_weights/part_0*
_output_shapes

:

÷
save/Assign_6AssignGinput_layer/input_layer/pvalue_level_embedding/embedding_weights/part_0save/RestoreV2:6*
_output_shapes

:
*Z
_classP
NLloc:@input_layer/input_layer/pvalue_level_embedding/embedding_weights/part_0*
T0
ű
save/Assign_7AssignIinput_layer/input_layer/shopping_level_embedding/embedding_weights/part_0save/RestoreV2:7*
T0*
_output_shapes

:
*\
_classR
PNloc:@input_layer/input_layer/shopping_level_embedding/embedding_weights/part_0
ď
save/Assign_8AssignBinput_layer/input_layer/user_id_embedding/embedding_weights/part_0save/RestoreV2:8*U
_classK
IGloc:@input_layer/input_layer/user_id_embedding/embedding_weights/part_0*
T0* 
_output_shapes
:
 
ů
save/Assign_9AssignGinput_layer/input_layer_1/adgroup_id_embedding/embedding_weights/part_0save/RestoreV2:9* 
_output_shapes
:
 *
T0*Z
_classP
NLloc:@input_layer/input_layer_1/adgroup_id_embedding/embedding_weights/part_0
ń
save/Assign_10AssignBinput_layer/input_layer_1/brand_embedding/embedding_weights/part_0save/RestoreV2:10*U
_classK
IGloc:@input_layer/input_layer_1/brand_embedding/embedding_weights/part_0*
T0* 
_output_shapes
:
 
ý
save/Assign_11AssignHinput_layer/input_layer_1/campaign_id_embedding/embedding_weights/part_0save/RestoreV2:11*
T0*[
_classQ
OMloc:@input_layer/input_layer_1/campaign_id_embedding/embedding_weights/part_0* 
_output_shapes
:
 
ô
save/Assign_12AssignDinput_layer/input_layer_1/cate_id_embedding/embedding_weights/part_0save/RestoreV2:12*
T0*
_output_shapes
:	N*W
_classM
KIloc:@input_layer/input_layer_1/cate_id_embedding/embedding_weights/part_0
÷
save/Assign_13AssignEinput_layer/input_layer_1/customer_embedding/embedding_weights/part_0save/RestoreV2:13*
T0* 
_output_shapes
:
 *X
_classN
LJloc:@input_layer/input_layer_1/customer_embedding/embedding_weights/part_0

save/Assign_14AssignNinput_layer/input_layer_1/final_gender_code_embedding/embedding_weights/part_0save/RestoreV2:14*
T0*
_output_shapes

:
*a
_classW
USloc:@input_layer/input_layer_1/final_gender_code_embedding/embedding_weights/part_0
ë
save/Assign_15Assign@input_layer/input_layer_1/pid_embedding/embedding_weights/part_0save/RestoreV2:15*S
_classI
GEloc:@input_layer/input_layer_1/pid_embedding/embedding_weights/part_0*
T0*
_output_shapes

:

ď
save/Assign_16AssignBinput_layer/input_layer_1/price_embedding/embedding_weights/part_0save/RestoreV2:16*
T0*
_output_shapes

:2*U
_classK
IGloc:@input_layer/input_layer_1/price_embedding/embedding_weights/part_0
ř
save/Assign_17AssignHitem_dnn_layer/item_dnn_layer/item_dnn_0/batch_normalization/beta/part_0save/RestoreV2:17*[
_classQ
OMloc:@item_dnn_layer/item_dnn_layer/item_dnn_0/batch_normalization/beta/part_0*
T0*
_output_shapes	
:
ú
save/Assign_18AssignIitem_dnn_layer/item_dnn_layer/item_dnn_0/batch_normalization/gamma/part_0save/RestoreV2:18*
T0*
_output_shapes	
:*\
_classR
PNloc:@item_dnn_layer/item_dnn_layer/item_dnn_0/batch_normalization/gamma/part_0
ř
save/Assign_19AssignHitem_dnn_layer/item_dnn_layer/item_dnn_0/batch_normalization/moving_meansave/RestoreV2:19*
T0*
_output_shapes	
:*[
_classQ
OMloc:@item_dnn_layer/item_dnn_layer/item_dnn_0/batch_normalization/moving_mean

save/Assign_20AssignLitem_dnn_layer/item_dnn_layer/item_dnn_0/batch_normalization/moving_variancesave/RestoreV2:20*
_output_shapes	
:*
T0*_
_classU
SQloc:@item_dnn_layer/item_dnn_layer/item_dnn_0/batch_normalization/moving_variance
Ü
save/Assign_21Assign:item_dnn_layer/item_dnn_layer/item_dnn_0/dense/bias/part_0save/RestoreV2:21*
T0*M
_classC
A?loc:@item_dnn_layer/item_dnn_layer/item_dnn_0/dense/bias/part_0*
_output_shapes	
:
ĺ
save/Assign_22Assign<item_dnn_layer/item_dnn_layer/item_dnn_0/dense/kernel/part_0save/RestoreV2:22*
T0* 
_output_shapes
:
*O
_classE
CAloc:@item_dnn_layer/item_dnn_layer/item_dnn_0/dense/kernel/part_0
ř
save/Assign_23AssignHitem_dnn_layer/item_dnn_layer/item_dnn_1/batch_normalization/beta/part_0save/RestoreV2:23*[
_classQ
OMloc:@item_dnn_layer/item_dnn_layer/item_dnn_1/batch_normalization/beta/part_0*
T0*
_output_shapes	
:
ú
save/Assign_24AssignIitem_dnn_layer/item_dnn_layer/item_dnn_1/batch_normalization/gamma/part_0save/RestoreV2:24*
T0*\
_classR
PNloc:@item_dnn_layer/item_dnn_layer/item_dnn_1/batch_normalization/gamma/part_0*
_output_shapes	
:
ř
save/Assign_25AssignHitem_dnn_layer/item_dnn_layer/item_dnn_1/batch_normalization/moving_meansave/RestoreV2:25*[
_classQ
OMloc:@item_dnn_layer/item_dnn_layer/item_dnn_1/batch_normalization/moving_mean*
_output_shapes	
:*
T0

save/Assign_26AssignLitem_dnn_layer/item_dnn_layer/item_dnn_1/batch_normalization/moving_variancesave/RestoreV2:26*
T0*
_output_shapes	
:*_
_classU
SQloc:@item_dnn_layer/item_dnn_layer/item_dnn_1/batch_normalization/moving_variance
Ü
save/Assign_27Assign:item_dnn_layer/item_dnn_layer/item_dnn_1/dense/bias/part_0save/RestoreV2:27*
_output_shapes	
:*
T0*M
_classC
A?loc:@item_dnn_layer/item_dnn_layer/item_dnn_1/dense/bias/part_0
ĺ
save/Assign_28Assign<item_dnn_layer/item_dnn_layer/item_dnn_1/dense/kernel/part_0save/RestoreV2:28*O
_classE
CAloc:@item_dnn_layer/item_dnn_layer/item_dnn_1/dense/kernel/part_0*
T0* 
_output_shapes
:

÷
save/Assign_29AssignHitem_dnn_layer/item_dnn_layer/item_dnn_2/batch_normalization/beta/part_0save/RestoreV2:29*[
_classQ
OMloc:@item_dnn_layer/item_dnn_layer/item_dnn_2/batch_normalization/beta/part_0*
_output_shapes
:@*
T0
ů
save/Assign_30AssignIitem_dnn_layer/item_dnn_layer/item_dnn_2/batch_normalization/gamma/part_0save/RestoreV2:30*
T0*
_output_shapes
:@*\
_classR
PNloc:@item_dnn_layer/item_dnn_layer/item_dnn_2/batch_normalization/gamma/part_0
÷
save/Assign_31AssignHitem_dnn_layer/item_dnn_layer/item_dnn_2/batch_normalization/moving_meansave/RestoreV2:31*
_output_shapes
:@*
T0*[
_classQ
OMloc:@item_dnn_layer/item_dnn_layer/item_dnn_2/batch_normalization/moving_mean
˙
save/Assign_32AssignLitem_dnn_layer/item_dnn_layer/item_dnn_2/batch_normalization/moving_variancesave/RestoreV2:32*
T0*
_output_shapes
:@*_
_classU
SQloc:@item_dnn_layer/item_dnn_layer/item_dnn_2/batch_normalization/moving_variance
Ű
save/Assign_33Assign:item_dnn_layer/item_dnn_layer/item_dnn_2/dense/bias/part_0save/RestoreV2:33*M
_classC
A?loc:@item_dnn_layer/item_dnn_layer/item_dnn_2/dense/bias/part_0*
_output_shapes
:@*
T0
ä
save/Assign_34Assign<item_dnn_layer/item_dnn_layer/item_dnn_2/dense/kernel/part_0save/RestoreV2:34*O
_classE
CAloc:@item_dnn_layer/item_dnn_layer/item_dnn_2/dense/kernel/part_0*
T0*
_output_shapes
:	@
Ű
save/Assign_35Assign:item_dnn_layer/item_dnn_layer/item_dnn_3/dense/bias/part_0save/RestoreV2:35*M
_classC
A?loc:@item_dnn_layer/item_dnn_layer/item_dnn_3/dense/bias/part_0*
T0*
_output_shapes
: 
ă
save/Assign_36Assign<item_dnn_layer/item_dnn_layer/item_dnn_3/dense/kernel/part_0save/RestoreV2:36*O
_classE
CAloc:@item_dnn_layer/item_dnn_layer/item_dnn_3/dense/kernel/part_0*
T0*
_output_shapes

:@ 
q
save/Assign_37Assignsim_bsave/RestoreV2:37*
_output_shapes
:*
T0*
_class

loc:@sim_b
u
save/Assign_38Assignsim_wsave/RestoreV2:38*
_output_shapes

:*
T0*
_class

loc:@sim_w
ř
save/Assign_39AssignHuser_dnn_layer/user_dnn_layer/user_dnn_0/batch_normalization/beta/part_0save/RestoreV2:39*
T0*
_output_shapes	
:*[
_classQ
OMloc:@user_dnn_layer/user_dnn_layer/user_dnn_0/batch_normalization/beta/part_0
ú
save/Assign_40AssignIuser_dnn_layer/user_dnn_layer/user_dnn_0/batch_normalization/gamma/part_0save/RestoreV2:40*
T0*
_output_shapes	
:*\
_classR
PNloc:@user_dnn_layer/user_dnn_layer/user_dnn_0/batch_normalization/gamma/part_0
ř
save/Assign_41AssignHuser_dnn_layer/user_dnn_layer/user_dnn_0/batch_normalization/moving_meansave/RestoreV2:41*
T0*
_output_shapes	
:*[
_classQ
OMloc:@user_dnn_layer/user_dnn_layer/user_dnn_0/batch_normalization/moving_mean

save/Assign_42AssignLuser_dnn_layer/user_dnn_layer/user_dnn_0/batch_normalization/moving_variancesave/RestoreV2:42*
T0*_
_classU
SQloc:@user_dnn_layer/user_dnn_layer/user_dnn_0/batch_normalization/moving_variance*
_output_shapes	
:
Ü
save/Assign_43Assign:user_dnn_layer/user_dnn_layer/user_dnn_0/dense/bias/part_0save/RestoreV2:43*
T0*
_output_shapes	
:*M
_classC
A?loc:@user_dnn_layer/user_dnn_layer/user_dnn_0/dense/bias/part_0
ĺ
save/Assign_44Assign<user_dnn_layer/user_dnn_layer/user_dnn_0/dense/kernel/part_0save/RestoreV2:44*
T0*O
_classE
CAloc:@user_dnn_layer/user_dnn_layer/user_dnn_0/dense/kernel/part_0* 
_output_shapes
:

ř
save/Assign_45AssignHuser_dnn_layer/user_dnn_layer/user_dnn_1/batch_normalization/beta/part_0save/RestoreV2:45*[
_classQ
OMloc:@user_dnn_layer/user_dnn_layer/user_dnn_1/batch_normalization/beta/part_0*
_output_shapes	
:*
T0
ú
save/Assign_46AssignIuser_dnn_layer/user_dnn_layer/user_dnn_1/batch_normalization/gamma/part_0save/RestoreV2:46*
_output_shapes	
:*
T0*\
_classR
PNloc:@user_dnn_layer/user_dnn_layer/user_dnn_1/batch_normalization/gamma/part_0
ř
save/Assign_47AssignHuser_dnn_layer/user_dnn_layer/user_dnn_1/batch_normalization/moving_meansave/RestoreV2:47*[
_classQ
OMloc:@user_dnn_layer/user_dnn_layer/user_dnn_1/batch_normalization/moving_mean*
_output_shapes	
:*
T0

save/Assign_48AssignLuser_dnn_layer/user_dnn_layer/user_dnn_1/batch_normalization/moving_variancesave/RestoreV2:48*
T0*
_output_shapes	
:*_
_classU
SQloc:@user_dnn_layer/user_dnn_layer/user_dnn_1/batch_normalization/moving_variance
Ü
save/Assign_49Assign:user_dnn_layer/user_dnn_layer/user_dnn_1/dense/bias/part_0save/RestoreV2:49*M
_classC
A?loc:@user_dnn_layer/user_dnn_layer/user_dnn_1/dense/bias/part_0*
_output_shapes	
:*
T0
ĺ
save/Assign_50Assign<user_dnn_layer/user_dnn_layer/user_dnn_1/dense/kernel/part_0save/RestoreV2:50* 
_output_shapes
:
*O
_classE
CAloc:@user_dnn_layer/user_dnn_layer/user_dnn_1/dense/kernel/part_0*
T0
÷
save/Assign_51AssignHuser_dnn_layer/user_dnn_layer/user_dnn_2/batch_normalization/beta/part_0save/RestoreV2:51*
T0*[
_classQ
OMloc:@user_dnn_layer/user_dnn_layer/user_dnn_2/batch_normalization/beta/part_0*
_output_shapes
:@
ů
save/Assign_52AssignIuser_dnn_layer/user_dnn_layer/user_dnn_2/batch_normalization/gamma/part_0save/RestoreV2:52*
_output_shapes
:@*\
_classR
PNloc:@user_dnn_layer/user_dnn_layer/user_dnn_2/batch_normalization/gamma/part_0*
T0
÷
save/Assign_53AssignHuser_dnn_layer/user_dnn_layer/user_dnn_2/batch_normalization/moving_meansave/RestoreV2:53*[
_classQ
OMloc:@user_dnn_layer/user_dnn_layer/user_dnn_2/batch_normalization/moving_mean*
T0*
_output_shapes
:@
˙
save/Assign_54AssignLuser_dnn_layer/user_dnn_layer/user_dnn_2/batch_normalization/moving_variancesave/RestoreV2:54*_
_classU
SQloc:@user_dnn_layer/user_dnn_layer/user_dnn_2/batch_normalization/moving_variance*
_output_shapes
:@*
T0
Ű
save/Assign_55Assign:user_dnn_layer/user_dnn_layer/user_dnn_2/dense/bias/part_0save/RestoreV2:55*
_output_shapes
:@*
T0*M
_classC
A?loc:@user_dnn_layer/user_dnn_layer/user_dnn_2/dense/bias/part_0
ä
save/Assign_56Assign<user_dnn_layer/user_dnn_layer/user_dnn_2/dense/kernel/part_0save/RestoreV2:56*O
_classE
CAloc:@user_dnn_layer/user_dnn_layer/user_dnn_2/dense/kernel/part_0*
T0*
_output_shapes
:	@
Ű
save/Assign_57Assign:user_dnn_layer/user_dnn_layer/user_dnn_3/dense/bias/part_0save/RestoreV2:57*
_output_shapes
: *M
_classC
A?loc:@user_dnn_layer/user_dnn_layer/user_dnn_3/dense/bias/part_0*
T0
ă
save/Assign_58Assign<user_dnn_layer/user_dnn_layer/user_dnn_3/dense/kernel/part_0save/RestoreV2:58*
T0*O
_classE
CAloc:@user_dnn_layer/user_dnn_layer/user_dnn_3/dense/kernel/part_0*
_output_shapes

:@ 
ů
save/restore_shardNoOp^save/Assign^save/Assign_1^save/Assign_10^save/Assign_11^save/Assign_12^save/Assign_13^save/Assign_14^save/Assign_15^save/Assign_16^save/Assign_17^save/Assign_18^save/Assign_19^save/Assign_2^save/Assign_20^save/Assign_21^save/Assign_22^save/Assign_23^save/Assign_24^save/Assign_25^save/Assign_26^save/Assign_27^save/Assign_28^save/Assign_29^save/Assign_3^save/Assign_30^save/Assign_31^save/Assign_32^save/Assign_33^save/Assign_34^save/Assign_35^save/Assign_36^save/Assign_37^save/Assign_38^save/Assign_39^save/Assign_4^save/Assign_40^save/Assign_41^save/Assign_42^save/Assign_43^save/Assign_44^save/Assign_45^save/Assign_46^save/Assign_47^save/Assign_48^save/Assign_49^save/Assign_5^save/Assign_50^save/Assign_51^save/Assign_52^save/Assign_53^save/Assign_54^save/Assign_55^save/Assign_56^save/Assign_57^save/Assign_58^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9
-
save/restore_allNoOp^save/restore_shard"<
save/Const:0save/Identity:0save/restore_all (5 @F8"m
global_step^\
Z
global_step:0global_step/Assignglobal_step/read:02global_step/Initializer/zeros:0H"ů
	summariesë
č
kuser_dnn_layer/user_dnn_layer/user_dnn_0/user_dnn_layer/user_dnn_layer/user_dnn_0/fraction_of_zero_values:0
^user_dnn_layer/user_dnn_layer/user_dnn_0/user_dnn_layer/user_dnn_layer/user_dnn_0/activation:0
kuser_dnn_layer/user_dnn_layer/user_dnn_1/user_dnn_layer/user_dnn_layer/user_dnn_1/fraction_of_zero_values:0
^user_dnn_layer/user_dnn_layer/user_dnn_1/user_dnn_layer/user_dnn_layer/user_dnn_1/activation:0
kuser_dnn_layer/user_dnn_layer/user_dnn_2/user_dnn_layer/user_dnn_layer/user_dnn_2/fraction_of_zero_values:0
^user_dnn_layer/user_dnn_layer/user_dnn_2/user_dnn_layer/user_dnn_layer/user_dnn_2/activation:0
kuser_dnn_layer/user_dnn_layer/user_dnn_3/user_dnn_layer/user_dnn_layer/user_dnn_3/fraction_of_zero_values:0
^user_dnn_layer/user_dnn_layer/user_dnn_3/user_dnn_layer/user_dnn_layer/user_dnn_3/activation:0
kitem_dnn_layer/item_dnn_layer/item_dnn_0/item_dnn_layer/item_dnn_layer/item_dnn_0/fraction_of_zero_values:0
^item_dnn_layer/item_dnn_layer/item_dnn_0/item_dnn_layer/item_dnn_layer/item_dnn_0/activation:0
kitem_dnn_layer/item_dnn_layer/item_dnn_1/item_dnn_layer/item_dnn_layer/item_dnn_1/fraction_of_zero_values:0
^item_dnn_layer/item_dnn_layer/item_dnn_1/item_dnn_layer/item_dnn_layer/item_dnn_1/activation:0
kitem_dnn_layer/item_dnn_layer/item_dnn_2/item_dnn_layer/item_dnn_layer/item_dnn_2/fraction_of_zero_values:0
^item_dnn_layer/item_dnn_layer/item_dnn_2/item_dnn_layer/item_dnn_layer/item_dnn_2/activation:0
kitem_dnn_layer/item_dnn_layer/item_dnn_3/item_dnn_layer/item_dnn_layer/item_dnn_3/fraction_of_zero_values:0
^item_dnn_layer/item_dnn_layer/item_dnn_3/item_dnn_layer/item_dnn_layer/item_dnn_3/activation:0"Ř4
model_variablesÄ4Á4

Finput_layer/input_layer/age_level_embedding/embedding_weights/part_0:0Kinput_layer/input_layer/age_level_embedding/embedding_weights/part_0/AssignKinput_layer/input_layer/age_level_embedding/embedding_weights/part_0/read:0"K
=input_layer/input_layer/age_level_embedding/embedding_weights
  "
2cinput_layer/input_layer/age_level_embedding/embedding_weights/part_0/Initializer/truncated_normal:08
Ľ
Iinput_layer/input_layer/cms_group_id_embedding/embedding_weights/part_0:0Ninput_layer/input_layer/cms_group_id_embedding/embedding_weights/part_0/AssignNinput_layer/input_layer/cms_group_id_embedding/embedding_weights/part_0/read:0"N
@input_layer/input_layer/cms_group_id_embedding/embedding_weightsd  "d2finput_layer/input_layer/cms_group_id_embedding/embedding_weights/part_0/Initializer/truncated_normal:08

Finput_layer/input_layer/cms_segid_embedding/embedding_weights/part_0:0Kinput_layer/input_layer/cms_segid_embedding/embedding_weights/part_0/AssignKinput_layer/input_layer/cms_segid_embedding/embedding_weights/part_0/read:0"K
=input_layer/input_layer/cms_segid_embedding/embedding_weightsd  "d2cinput_layer/input_layer/cms_segid_embedding/embedding_weights/part_0/Initializer/truncated_normal:08
Í
Qinput_layer/input_layer/new_user_class_level_embedding/embedding_weights/part_0:0Vinput_layer/input_layer/new_user_class_level_embedding/embedding_weights/part_0/AssignVinput_layer/input_layer/new_user_class_level_embedding/embedding_weights/part_0/read:0"V
Hinput_layer/input_layer/new_user_class_level_embedding/embedding_weights
  "
2ninput_layer/input_layer/new_user_class_level_embedding/embedding_weights/part_0/Initializer/truncated_normal:08

Ginput_layer/input_layer/occupation_embedding/embedding_weights/part_0:0Linput_layer/input_layer/occupation_embedding/embedding_weights/part_0/AssignLinput_layer/input_layer/occupation_embedding/embedding_weights/part_0/read:0"L
>input_layer/input_layer/occupation_embedding/embedding_weights
  "
2dinput_layer/input_layer/occupation_embedding/embedding_weights/part_0/Initializer/truncated_normal:08
Ľ
Iinput_layer/input_layer/pvalue_level_embedding/embedding_weights/part_0:0Ninput_layer/input_layer/pvalue_level_embedding/embedding_weights/part_0/AssignNinput_layer/input_layer/pvalue_level_embedding/embedding_weights/part_0/read:0"N
@input_layer/input_layer/pvalue_level_embedding/embedding_weights
  "
2finput_layer/input_layer/pvalue_level_embedding/embedding_weights/part_0/Initializer/truncated_normal:08
Ż
Kinput_layer/input_layer/shopping_level_embedding/embedding_weights/part_0:0Pinput_layer/input_layer/shopping_level_embedding/embedding_weights/part_0/AssignPinput_layer/input_layer/shopping_level_embedding/embedding_weights/part_0/read:0"P
Binput_layer/input_layer/shopping_level_embedding/embedding_weights
  "
2hinput_layer/input_layer/shopping_level_embedding/embedding_weights/part_0/Initializer/truncated_normal:08

Dinput_layer/input_layer/user_id_embedding/embedding_weights/part_0:0Iinput_layer/input_layer/user_id_embedding/embedding_weights/part_0/AssignIinput_layer/input_layer/user_id_embedding/embedding_weights/part_0/read:0"M
;input_layer/input_layer/user_id_embedding/embedding_weights   " 2ainput_layer/input_layer/user_id_embedding/embedding_weights/part_0/Initializer/truncated_normal:08
Š
Iinput_layer/input_layer_1/adgroup_id_embedding/embedding_weights/part_0:0Ninput_layer/input_layer_1/adgroup_id_embedding/embedding_weights/part_0/AssignNinput_layer/input_layer_1/adgroup_id_embedding/embedding_weights/part_0/read:0"R
@input_layer/input_layer_1/adgroup_id_embedding/embedding_weights   " 2finput_layer/input_layer_1/adgroup_id_embedding/embedding_weights/part_0/Initializer/truncated_normal:08

Dinput_layer/input_layer_1/brand_embedding/embedding_weights/part_0:0Iinput_layer/input_layer_1/brand_embedding/embedding_weights/part_0/AssignIinput_layer/input_layer_1/brand_embedding/embedding_weights/part_0/read:0"M
;input_layer/input_layer_1/brand_embedding/embedding_weights   " 2ainput_layer/input_layer_1/brand_embedding/embedding_weights/part_0/Initializer/truncated_normal:08
Ž
Jinput_layer/input_layer_1/campaign_id_embedding/embedding_weights/part_0:0Oinput_layer/input_layer_1/campaign_id_embedding/embedding_weights/part_0/AssignOinput_layer/input_layer_1/campaign_id_embedding/embedding_weights/part_0/read:0"S
Ainput_layer/input_layer_1/campaign_id_embedding/embedding_weights   " 2ginput_layer/input_layer_1/campaign_id_embedding/embedding_weights/part_0/Initializer/truncated_normal:08

Finput_layer/input_layer_1/cate_id_embedding/embedding_weights/part_0:0Kinput_layer/input_layer_1/cate_id_embedding/embedding_weights/part_0/AssignKinput_layer/input_layer_1/cate_id_embedding/embedding_weights/part_0/read:0"M
=input_layer/input_layer_1/cate_id_embedding/embedding_weightsN  "N2cinput_layer/input_layer_1/cate_id_embedding/embedding_weights/part_0/Initializer/truncated_normal:08

Ginput_layer/input_layer_1/customer_embedding/embedding_weights/part_0:0Linput_layer/input_layer_1/customer_embedding/embedding_weights/part_0/AssignLinput_layer/input_layer_1/customer_embedding/embedding_weights/part_0/read:0"P
>input_layer/input_layer_1/customer_embedding/embedding_weights   " 2dinput_layer/input_layer_1/customer_embedding/embedding_weights/part_0/Initializer/truncated_normal:08
Č
Pinput_layer/input_layer_1/final_gender_code_embedding/embedding_weights/part_0:0Uinput_layer/input_layer_1/final_gender_code_embedding/embedding_weights/part_0/AssignUinput_layer/input_layer_1/final_gender_code_embedding/embedding_weights/part_0/read:0"U
Ginput_layer/input_layer_1/final_gender_code_embedding/embedding_weights
  "
2minput_layer/input_layer_1/final_gender_code_embedding/embedding_weights/part_0/Initializer/truncated_normal:08

Binput_layer/input_layer_1/pid_embedding/embedding_weights/part_0:0Ginput_layer/input_layer_1/pid_embedding/embedding_weights/part_0/AssignGinput_layer/input_layer_1/pid_embedding/embedding_weights/part_0/read:0"G
9input_layer/input_layer_1/pid_embedding/embedding_weights
  "
2_input_layer/input_layer_1/pid_embedding/embedding_weights/part_0/Initializer/truncated_normal:08

Dinput_layer/input_layer_1/price_embedding/embedding_weights/part_0:0Iinput_layer/input_layer_1/price_embedding/embedding_weights/part_0/AssignIinput_layer/input_layer_1/price_embedding/embedding_weights/part_0/read:0"I
;input_layer/input_layer_1/price_embedding/embedding_weights2  "22ainput_layer/input_layer_1/price_embedding/embedding_weights/part_0/Initializer/truncated_normal:08"Ć

update_opsˇ
´
Luser_dnn_layer/user_dnn_layer/user_dnn_0/batch_normalization/AssignMovingAvg
Nuser_dnn_layer/user_dnn_layer/user_dnn_0/batch_normalization/AssignMovingAvg_1
Luser_dnn_layer/user_dnn_layer/user_dnn_1/batch_normalization/AssignMovingAvg
Nuser_dnn_layer/user_dnn_layer/user_dnn_1/batch_normalization/AssignMovingAvg_1
Luser_dnn_layer/user_dnn_layer/user_dnn_2/batch_normalization/AssignMovingAvg
Nuser_dnn_layer/user_dnn_layer/user_dnn_2/batch_normalization/AssignMovingAvg_1
Litem_dnn_layer/item_dnn_layer/item_dnn_0/batch_normalization/AssignMovingAvg
Nitem_dnn_layer/item_dnn_layer/item_dnn_0/batch_normalization/AssignMovingAvg_1
Litem_dnn_layer/item_dnn_layer/item_dnn_1/batch_normalization/AssignMovingAvg
Nitem_dnn_layer/item_dnn_layer/item_dnn_1/batch_normalization/AssignMovingAvg_1
Litem_dnn_layer/item_dnn_layer/item_dnn_2/batch_normalization/AssignMovingAvg
Nitem_dnn_layer/item_dnn_layer/item_dnn_2/batch_normalization/AssignMovingAvg_1"Ý
trainable_variablesÄŔ

Finput_layer/input_layer/age_level_embedding/embedding_weights/part_0:0Kinput_layer/input_layer/age_level_embedding/embedding_weights/part_0/AssignKinput_layer/input_layer/age_level_embedding/embedding_weights/part_0/read:0"K
=input_layer/input_layer/age_level_embedding/embedding_weights
  "
2cinput_layer/input_layer/age_level_embedding/embedding_weights/part_0/Initializer/truncated_normal:08
Ľ
Iinput_layer/input_layer/cms_group_id_embedding/embedding_weights/part_0:0Ninput_layer/input_layer/cms_group_id_embedding/embedding_weights/part_0/AssignNinput_layer/input_layer/cms_group_id_embedding/embedding_weights/part_0/read:0"N
@input_layer/input_layer/cms_group_id_embedding/embedding_weightsd  "d2finput_layer/input_layer/cms_group_id_embedding/embedding_weights/part_0/Initializer/truncated_normal:08

Finput_layer/input_layer/cms_segid_embedding/embedding_weights/part_0:0Kinput_layer/input_layer/cms_segid_embedding/embedding_weights/part_0/AssignKinput_layer/input_layer/cms_segid_embedding/embedding_weights/part_0/read:0"K
=input_layer/input_layer/cms_segid_embedding/embedding_weightsd  "d2cinput_layer/input_layer/cms_segid_embedding/embedding_weights/part_0/Initializer/truncated_normal:08
Í
Qinput_layer/input_layer/new_user_class_level_embedding/embedding_weights/part_0:0Vinput_layer/input_layer/new_user_class_level_embedding/embedding_weights/part_0/AssignVinput_layer/input_layer/new_user_class_level_embedding/embedding_weights/part_0/read:0"V
Hinput_layer/input_layer/new_user_class_level_embedding/embedding_weights
  "
2ninput_layer/input_layer/new_user_class_level_embedding/embedding_weights/part_0/Initializer/truncated_normal:08

Ginput_layer/input_layer/occupation_embedding/embedding_weights/part_0:0Linput_layer/input_layer/occupation_embedding/embedding_weights/part_0/AssignLinput_layer/input_layer/occupation_embedding/embedding_weights/part_0/read:0"L
>input_layer/input_layer/occupation_embedding/embedding_weights
  "
2dinput_layer/input_layer/occupation_embedding/embedding_weights/part_0/Initializer/truncated_normal:08
Ľ
Iinput_layer/input_layer/pvalue_level_embedding/embedding_weights/part_0:0Ninput_layer/input_layer/pvalue_level_embedding/embedding_weights/part_0/AssignNinput_layer/input_layer/pvalue_level_embedding/embedding_weights/part_0/read:0"N
@input_layer/input_layer/pvalue_level_embedding/embedding_weights
  "
2finput_layer/input_layer/pvalue_level_embedding/embedding_weights/part_0/Initializer/truncated_normal:08
Ż
Kinput_layer/input_layer/shopping_level_embedding/embedding_weights/part_0:0Pinput_layer/input_layer/shopping_level_embedding/embedding_weights/part_0/AssignPinput_layer/input_layer/shopping_level_embedding/embedding_weights/part_0/read:0"P
Binput_layer/input_layer/shopping_level_embedding/embedding_weights
  "
2hinput_layer/input_layer/shopping_level_embedding/embedding_weights/part_0/Initializer/truncated_normal:08

Dinput_layer/input_layer/user_id_embedding/embedding_weights/part_0:0Iinput_layer/input_layer/user_id_embedding/embedding_weights/part_0/AssignIinput_layer/input_layer/user_id_embedding/embedding_weights/part_0/read:0"M
;input_layer/input_layer/user_id_embedding/embedding_weights   " 2ainput_layer/input_layer/user_id_embedding/embedding_weights/part_0/Initializer/truncated_normal:08
Š
Iinput_layer/input_layer_1/adgroup_id_embedding/embedding_weights/part_0:0Ninput_layer/input_layer_1/adgroup_id_embedding/embedding_weights/part_0/AssignNinput_layer/input_layer_1/adgroup_id_embedding/embedding_weights/part_0/read:0"R
@input_layer/input_layer_1/adgroup_id_embedding/embedding_weights   " 2finput_layer/input_layer_1/adgroup_id_embedding/embedding_weights/part_0/Initializer/truncated_normal:08

Dinput_layer/input_layer_1/brand_embedding/embedding_weights/part_0:0Iinput_layer/input_layer_1/brand_embedding/embedding_weights/part_0/AssignIinput_layer/input_layer_1/brand_embedding/embedding_weights/part_0/read:0"M
;input_layer/input_layer_1/brand_embedding/embedding_weights   " 2ainput_layer/input_layer_1/brand_embedding/embedding_weights/part_0/Initializer/truncated_normal:08
Ž
Jinput_layer/input_layer_1/campaign_id_embedding/embedding_weights/part_0:0Oinput_layer/input_layer_1/campaign_id_embedding/embedding_weights/part_0/AssignOinput_layer/input_layer_1/campaign_id_embedding/embedding_weights/part_0/read:0"S
Ainput_layer/input_layer_1/campaign_id_embedding/embedding_weights   " 2ginput_layer/input_layer_1/campaign_id_embedding/embedding_weights/part_0/Initializer/truncated_normal:08

Finput_layer/input_layer_1/cate_id_embedding/embedding_weights/part_0:0Kinput_layer/input_layer_1/cate_id_embedding/embedding_weights/part_0/AssignKinput_layer/input_layer_1/cate_id_embedding/embedding_weights/part_0/read:0"M
=input_layer/input_layer_1/cate_id_embedding/embedding_weightsN  "N2cinput_layer/input_layer_1/cate_id_embedding/embedding_weights/part_0/Initializer/truncated_normal:08

Ginput_layer/input_layer_1/customer_embedding/embedding_weights/part_0:0Linput_layer/input_layer_1/customer_embedding/embedding_weights/part_0/AssignLinput_layer/input_layer_1/customer_embedding/embedding_weights/part_0/read:0"P
>input_layer/input_layer_1/customer_embedding/embedding_weights   " 2dinput_layer/input_layer_1/customer_embedding/embedding_weights/part_0/Initializer/truncated_normal:08
Č
Pinput_layer/input_layer_1/final_gender_code_embedding/embedding_weights/part_0:0Uinput_layer/input_layer_1/final_gender_code_embedding/embedding_weights/part_0/AssignUinput_layer/input_layer_1/final_gender_code_embedding/embedding_weights/part_0/read:0"U
Ginput_layer/input_layer_1/final_gender_code_embedding/embedding_weights
  "
2minput_layer/input_layer_1/final_gender_code_embedding/embedding_weights/part_0/Initializer/truncated_normal:08

Binput_layer/input_layer_1/pid_embedding/embedding_weights/part_0:0Ginput_layer/input_layer_1/pid_embedding/embedding_weights/part_0/AssignGinput_layer/input_layer_1/pid_embedding/embedding_weights/part_0/read:0"G
9input_layer/input_layer_1/pid_embedding/embedding_weights
  "
2_input_layer/input_layer_1/pid_embedding/embedding_weights/part_0/Initializer/truncated_normal:08

Dinput_layer/input_layer_1/price_embedding/embedding_weights/part_0:0Iinput_layer/input_layer_1/price_embedding/embedding_weights/part_0/AssignIinput_layer/input_layer_1/price_embedding/embedding_weights/part_0/read:0"I
;input_layer/input_layer_1/price_embedding/embedding_weights2  "22ainput_layer/input_layer_1/price_embedding/embedding_weights/part_0/Initializer/truncated_normal:08
đ
>user_dnn_layer/user_dnn_layer/user_dnn_0/dense/kernel/part_0:0Cuser_dnn_layer/user_dnn_layer/user_dnn_0/dense/kernel/part_0/AssignCuser_dnn_layer/user_dnn_layer/user_dnn_0/dense/kernel/part_0/read:0"G
5user_dnn_layer/user_dnn_layer/user_dnn_0/dense/kernel  "2Yuser_dnn_layer/user_dnn_layer/user_dnn_0/dense/kernel/part_0/Initializer/random_uniform:08
Ř
<user_dnn_layer/user_dnn_layer/user_dnn_0/dense/bias/part_0:0Auser_dnn_layer/user_dnn_layer/user_dnn_0/dense/bias/part_0/AssignAuser_dnn_layer/user_dnn_layer/user_dnn_0/dense/bias/part_0/read:0"@
3user_dnn_layer/user_dnn_layer/user_dnn_0/dense/bias "2Nuser_dnn_layer/user_dnn_layer/user_dnn_0/dense/bias/part_0/Initializer/zeros:08
˘
Kuser_dnn_layer/user_dnn_layer/user_dnn_0/batch_normalization/gamma/part_0:0Puser_dnn_layer/user_dnn_layer/user_dnn_0/batch_normalization/gamma/part_0/AssignPuser_dnn_layer/user_dnn_layer/user_dnn_0/batch_normalization/gamma/part_0/read:0"O
Buser_dnn_layer/user_dnn_layer/user_dnn_0/batch_normalization/gamma "2\user_dnn_layer/user_dnn_layer/user_dnn_0/batch_normalization/gamma/part_0/Initializer/ones:08

Juser_dnn_layer/user_dnn_layer/user_dnn_0/batch_normalization/beta/part_0:0Ouser_dnn_layer/user_dnn_layer/user_dnn_0/batch_normalization/beta/part_0/AssignOuser_dnn_layer/user_dnn_layer/user_dnn_0/batch_normalization/beta/part_0/read:0"N
Auser_dnn_layer/user_dnn_layer/user_dnn_0/batch_normalization/beta "2\user_dnn_layer/user_dnn_layer/user_dnn_0/batch_normalization/beta/part_0/Initializer/zeros:08
đ
>user_dnn_layer/user_dnn_layer/user_dnn_1/dense/kernel/part_0:0Cuser_dnn_layer/user_dnn_layer/user_dnn_1/dense/kernel/part_0/AssignCuser_dnn_layer/user_dnn_layer/user_dnn_1/dense/kernel/part_0/read:0"G
5user_dnn_layer/user_dnn_layer/user_dnn_1/dense/kernel  "2Yuser_dnn_layer/user_dnn_layer/user_dnn_1/dense/kernel/part_0/Initializer/random_uniform:08
Ř
<user_dnn_layer/user_dnn_layer/user_dnn_1/dense/bias/part_0:0Auser_dnn_layer/user_dnn_layer/user_dnn_1/dense/bias/part_0/AssignAuser_dnn_layer/user_dnn_layer/user_dnn_1/dense/bias/part_0/read:0"@
3user_dnn_layer/user_dnn_layer/user_dnn_1/dense/bias "2Nuser_dnn_layer/user_dnn_layer/user_dnn_1/dense/bias/part_0/Initializer/zeros:08
˘
Kuser_dnn_layer/user_dnn_layer/user_dnn_1/batch_normalization/gamma/part_0:0Puser_dnn_layer/user_dnn_layer/user_dnn_1/batch_normalization/gamma/part_0/AssignPuser_dnn_layer/user_dnn_layer/user_dnn_1/batch_normalization/gamma/part_0/read:0"O
Buser_dnn_layer/user_dnn_layer/user_dnn_1/batch_normalization/gamma "2\user_dnn_layer/user_dnn_layer/user_dnn_1/batch_normalization/gamma/part_0/Initializer/ones:08

Juser_dnn_layer/user_dnn_layer/user_dnn_1/batch_normalization/beta/part_0:0Ouser_dnn_layer/user_dnn_layer/user_dnn_1/batch_normalization/beta/part_0/AssignOuser_dnn_layer/user_dnn_layer/user_dnn_1/batch_normalization/beta/part_0/read:0"N
Auser_dnn_layer/user_dnn_layer/user_dnn_1/batch_normalization/beta "2\user_dnn_layer/user_dnn_layer/user_dnn_1/batch_normalization/beta/part_0/Initializer/zeros:08
î
>user_dnn_layer/user_dnn_layer/user_dnn_2/dense/kernel/part_0:0Cuser_dnn_layer/user_dnn_layer/user_dnn_2/dense/kernel/part_0/AssignCuser_dnn_layer/user_dnn_layer/user_dnn_2/dense/kernel/part_0/read:0"E
5user_dnn_layer/user_dnn_layer/user_dnn_2/dense/kernel@  "@2Yuser_dnn_layer/user_dnn_layer/user_dnn_2/dense/kernel/part_0/Initializer/random_uniform:08
Ö
<user_dnn_layer/user_dnn_layer/user_dnn_2/dense/bias/part_0:0Auser_dnn_layer/user_dnn_layer/user_dnn_2/dense/bias/part_0/AssignAuser_dnn_layer/user_dnn_layer/user_dnn_2/dense/bias/part_0/read:0">
3user_dnn_layer/user_dnn_layer/user_dnn_2/dense/bias@ "@2Nuser_dnn_layer/user_dnn_layer/user_dnn_2/dense/bias/part_0/Initializer/zeros:08
 
Kuser_dnn_layer/user_dnn_layer/user_dnn_2/batch_normalization/gamma/part_0:0Puser_dnn_layer/user_dnn_layer/user_dnn_2/batch_normalization/gamma/part_0/AssignPuser_dnn_layer/user_dnn_layer/user_dnn_2/batch_normalization/gamma/part_0/read:0"M
Buser_dnn_layer/user_dnn_layer/user_dnn_2/batch_normalization/gamma@ "@2\user_dnn_layer/user_dnn_layer/user_dnn_2/batch_normalization/gamma/part_0/Initializer/ones:08

Juser_dnn_layer/user_dnn_layer/user_dnn_2/batch_normalization/beta/part_0:0Ouser_dnn_layer/user_dnn_layer/user_dnn_2/batch_normalization/beta/part_0/AssignOuser_dnn_layer/user_dnn_layer/user_dnn_2/batch_normalization/beta/part_0/read:0"L
Auser_dnn_layer/user_dnn_layer/user_dnn_2/batch_normalization/beta@ "@2\user_dnn_layer/user_dnn_layer/user_dnn_2/batch_normalization/beta/part_0/Initializer/zeros:08
ě
>user_dnn_layer/user_dnn_layer/user_dnn_3/dense/kernel/part_0:0Cuser_dnn_layer/user_dnn_layer/user_dnn_3/dense/kernel/part_0/AssignCuser_dnn_layer/user_dnn_layer/user_dnn_3/dense/kernel/part_0/read:0"C
5user_dnn_layer/user_dnn_layer/user_dnn_3/dense/kernel@   "@ 2Yuser_dnn_layer/user_dnn_layer/user_dnn_3/dense/kernel/part_0/Initializer/random_uniform:08
Ö
<user_dnn_layer/user_dnn_layer/user_dnn_3/dense/bias/part_0:0Auser_dnn_layer/user_dnn_layer/user_dnn_3/dense/bias/part_0/AssignAuser_dnn_layer/user_dnn_layer/user_dnn_3/dense/bias/part_0/read:0">
3user_dnn_layer/user_dnn_layer/user_dnn_3/dense/bias  " 2Nuser_dnn_layer/user_dnn_layer/user_dnn_3/dense/bias/part_0/Initializer/zeros:08
đ
>item_dnn_layer/item_dnn_layer/item_dnn_0/dense/kernel/part_0:0Citem_dnn_layer/item_dnn_layer/item_dnn_0/dense/kernel/part_0/AssignCitem_dnn_layer/item_dnn_layer/item_dnn_0/dense/kernel/part_0/read:0"G
5item_dnn_layer/item_dnn_layer/item_dnn_0/dense/kernel  "2Yitem_dnn_layer/item_dnn_layer/item_dnn_0/dense/kernel/part_0/Initializer/random_uniform:08
Ř
<item_dnn_layer/item_dnn_layer/item_dnn_0/dense/bias/part_0:0Aitem_dnn_layer/item_dnn_layer/item_dnn_0/dense/bias/part_0/AssignAitem_dnn_layer/item_dnn_layer/item_dnn_0/dense/bias/part_0/read:0"@
3item_dnn_layer/item_dnn_layer/item_dnn_0/dense/bias "2Nitem_dnn_layer/item_dnn_layer/item_dnn_0/dense/bias/part_0/Initializer/zeros:08
˘
Kitem_dnn_layer/item_dnn_layer/item_dnn_0/batch_normalization/gamma/part_0:0Pitem_dnn_layer/item_dnn_layer/item_dnn_0/batch_normalization/gamma/part_0/AssignPitem_dnn_layer/item_dnn_layer/item_dnn_0/batch_normalization/gamma/part_0/read:0"O
Bitem_dnn_layer/item_dnn_layer/item_dnn_0/batch_normalization/gamma "2\item_dnn_layer/item_dnn_layer/item_dnn_0/batch_normalization/gamma/part_0/Initializer/ones:08

Jitem_dnn_layer/item_dnn_layer/item_dnn_0/batch_normalization/beta/part_0:0Oitem_dnn_layer/item_dnn_layer/item_dnn_0/batch_normalization/beta/part_0/AssignOitem_dnn_layer/item_dnn_layer/item_dnn_0/batch_normalization/beta/part_0/read:0"N
Aitem_dnn_layer/item_dnn_layer/item_dnn_0/batch_normalization/beta "2\item_dnn_layer/item_dnn_layer/item_dnn_0/batch_normalization/beta/part_0/Initializer/zeros:08
đ
>item_dnn_layer/item_dnn_layer/item_dnn_1/dense/kernel/part_0:0Citem_dnn_layer/item_dnn_layer/item_dnn_1/dense/kernel/part_0/AssignCitem_dnn_layer/item_dnn_layer/item_dnn_1/dense/kernel/part_0/read:0"G
5item_dnn_layer/item_dnn_layer/item_dnn_1/dense/kernel  "2Yitem_dnn_layer/item_dnn_layer/item_dnn_1/dense/kernel/part_0/Initializer/random_uniform:08
Ř
<item_dnn_layer/item_dnn_layer/item_dnn_1/dense/bias/part_0:0Aitem_dnn_layer/item_dnn_layer/item_dnn_1/dense/bias/part_0/AssignAitem_dnn_layer/item_dnn_layer/item_dnn_1/dense/bias/part_0/read:0"@
3item_dnn_layer/item_dnn_layer/item_dnn_1/dense/bias "2Nitem_dnn_layer/item_dnn_layer/item_dnn_1/dense/bias/part_0/Initializer/zeros:08
˘
Kitem_dnn_layer/item_dnn_layer/item_dnn_1/batch_normalization/gamma/part_0:0Pitem_dnn_layer/item_dnn_layer/item_dnn_1/batch_normalization/gamma/part_0/AssignPitem_dnn_layer/item_dnn_layer/item_dnn_1/batch_normalization/gamma/part_0/read:0"O
Bitem_dnn_layer/item_dnn_layer/item_dnn_1/batch_normalization/gamma "2\item_dnn_layer/item_dnn_layer/item_dnn_1/batch_normalization/gamma/part_0/Initializer/ones:08

Jitem_dnn_layer/item_dnn_layer/item_dnn_1/batch_normalization/beta/part_0:0Oitem_dnn_layer/item_dnn_layer/item_dnn_1/batch_normalization/beta/part_0/AssignOitem_dnn_layer/item_dnn_layer/item_dnn_1/batch_normalization/beta/part_0/read:0"N
Aitem_dnn_layer/item_dnn_layer/item_dnn_1/batch_normalization/beta "2\item_dnn_layer/item_dnn_layer/item_dnn_1/batch_normalization/beta/part_0/Initializer/zeros:08
î
>item_dnn_layer/item_dnn_layer/item_dnn_2/dense/kernel/part_0:0Citem_dnn_layer/item_dnn_layer/item_dnn_2/dense/kernel/part_0/AssignCitem_dnn_layer/item_dnn_layer/item_dnn_2/dense/kernel/part_0/read:0"E
5item_dnn_layer/item_dnn_layer/item_dnn_2/dense/kernel@  "@2Yitem_dnn_layer/item_dnn_layer/item_dnn_2/dense/kernel/part_0/Initializer/random_uniform:08
Ö
<item_dnn_layer/item_dnn_layer/item_dnn_2/dense/bias/part_0:0Aitem_dnn_layer/item_dnn_layer/item_dnn_2/dense/bias/part_0/AssignAitem_dnn_layer/item_dnn_layer/item_dnn_2/dense/bias/part_0/read:0">
3item_dnn_layer/item_dnn_layer/item_dnn_2/dense/bias@ "@2Nitem_dnn_layer/item_dnn_layer/item_dnn_2/dense/bias/part_0/Initializer/zeros:08
 
Kitem_dnn_layer/item_dnn_layer/item_dnn_2/batch_normalization/gamma/part_0:0Pitem_dnn_layer/item_dnn_layer/item_dnn_2/batch_normalization/gamma/part_0/AssignPitem_dnn_layer/item_dnn_layer/item_dnn_2/batch_normalization/gamma/part_0/read:0"M
Bitem_dnn_layer/item_dnn_layer/item_dnn_2/batch_normalization/gamma@ "@2\item_dnn_layer/item_dnn_layer/item_dnn_2/batch_normalization/gamma/part_0/Initializer/ones:08

Jitem_dnn_layer/item_dnn_layer/item_dnn_2/batch_normalization/beta/part_0:0Oitem_dnn_layer/item_dnn_layer/item_dnn_2/batch_normalization/beta/part_0/AssignOitem_dnn_layer/item_dnn_layer/item_dnn_2/batch_normalization/beta/part_0/read:0"L
Aitem_dnn_layer/item_dnn_layer/item_dnn_2/batch_normalization/beta@ "@2\item_dnn_layer/item_dnn_layer/item_dnn_2/batch_normalization/beta/part_0/Initializer/zeros:08
ě
>item_dnn_layer/item_dnn_layer/item_dnn_3/dense/kernel/part_0:0Citem_dnn_layer/item_dnn_layer/item_dnn_3/dense/kernel/part_0/AssignCitem_dnn_layer/item_dnn_layer/item_dnn_3/dense/kernel/part_0/read:0"C
5item_dnn_layer/item_dnn_layer/item_dnn_3/dense/kernel@   "@ 2Yitem_dnn_layer/item_dnn_layer/item_dnn_3/dense/kernel/part_0/Initializer/random_uniform:08
Ö
<item_dnn_layer/item_dnn_layer/item_dnn_3/dense/bias/part_0:0Aitem_dnn_layer/item_dnn_layer/item_dnn_3/dense/bias/part_0/AssignAitem_dnn_layer/item_dnn_layer/item_dnn_3/dense/bias/part_0/read:0">
3item_dnn_layer/item_dnn_layer/item_dnn_3/dense/bias  " 2Nitem_dnn_layer/item_dnn_layer/item_dnn_3/dense/bias/part_0/Initializer/zeros:08
A
sim_w:0sim_w/Assignsim_w/read:02sim_w/Initializer/ones:08
B
sim_b:0sim_b/Assignsim_b/read:02sim_b/Initializer/zeros:08"Î
cond_contextź¸


Euser_dnn_layer/user_dnn_layer/user_dnn_0/zero_fraction/cond/cond_textEuser_dnn_layer/user_dnn_layer/user_dnn_0/zero_fraction/cond/pred_id:0Fuser_dnn_layer/user_dnn_layer/user_dnn_0/zero_fraction/cond/switch_t:0 *Ź
/user_dnn_layer/user_dnn_layer/user_dnn_0/Relu:0
Buser_dnn_layer/user_dnn_layer/user_dnn_0/zero_fraction/cond/Cast:0
Puser_dnn_layer/user_dnn_layer/user_dnn_0/zero_fraction/cond/count_nonzero/Cast:0
Quser_dnn_layer/user_dnn_layer/user_dnn_0/zero_fraction/cond/count_nonzero/Const:0
[user_dnn_layer/user_dnn_layer/user_dnn_0/zero_fraction/cond/count_nonzero/NotEqual/Switch:1
Tuser_dnn_layer/user_dnn_layer/user_dnn_0/zero_fraction/cond/count_nonzero/NotEqual:0
Yuser_dnn_layer/user_dnn_layer/user_dnn_0/zero_fraction/cond/count_nonzero/nonzero_count:0
Quser_dnn_layer/user_dnn_layer/user_dnn_0/zero_fraction/cond/count_nonzero/zeros:0
Euser_dnn_layer/user_dnn_layer/user_dnn_0/zero_fraction/cond/pred_id:0
Fuser_dnn_layer/user_dnn_layer/user_dnn_0/zero_fraction/cond/switch_t:0
Euser_dnn_layer/user_dnn_layer/user_dnn_0/zero_fraction/cond/pred_id:0Euser_dnn_layer/user_dnn_layer/user_dnn_0/zero_fraction/cond/pred_id:0
/user_dnn_layer/user_dnn_layer/user_dnn_0/Relu:0[user_dnn_layer/user_dnn_layer/user_dnn_0/zero_fraction/cond/count_nonzero/NotEqual/Switch:1
Ń	
Guser_dnn_layer/user_dnn_layer/user_dnn_0/zero_fraction/cond/cond_text_1Euser_dnn_layer/user_dnn_layer/user_dnn_0/zero_fraction/cond/pred_id:0Fuser_dnn_layer/user_dnn_layer/user_dnn_0/zero_fraction/cond/switch_f:0*ö
/user_dnn_layer/user_dnn_layer/user_dnn_0/Relu:0
Ruser_dnn_layer/user_dnn_layer/user_dnn_0/zero_fraction/cond/count_nonzero_1/Cast:0
Suser_dnn_layer/user_dnn_layer/user_dnn_0/zero_fraction/cond/count_nonzero_1/Const:0
]user_dnn_layer/user_dnn_layer/user_dnn_0/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0
Vuser_dnn_layer/user_dnn_layer/user_dnn_0/zero_fraction/cond/count_nonzero_1/NotEqual:0
[user_dnn_layer/user_dnn_layer/user_dnn_0/zero_fraction/cond/count_nonzero_1/nonzero_count:0
Suser_dnn_layer/user_dnn_layer/user_dnn_0/zero_fraction/cond/count_nonzero_1/zeros:0
Euser_dnn_layer/user_dnn_layer/user_dnn_0/zero_fraction/cond/pred_id:0
Fuser_dnn_layer/user_dnn_layer/user_dnn_0/zero_fraction/cond/switch_f:0
Euser_dnn_layer/user_dnn_layer/user_dnn_0/zero_fraction/cond/pred_id:0Euser_dnn_layer/user_dnn_layer/user_dnn_0/zero_fraction/cond/pred_id:0
/user_dnn_layer/user_dnn_layer/user_dnn_0/Relu:0]user_dnn_layer/user_dnn_layer/user_dnn_0/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0


Euser_dnn_layer/user_dnn_layer/user_dnn_1/zero_fraction/cond/cond_textEuser_dnn_layer/user_dnn_layer/user_dnn_1/zero_fraction/cond/pred_id:0Fuser_dnn_layer/user_dnn_layer/user_dnn_1/zero_fraction/cond/switch_t:0 *Ź
/user_dnn_layer/user_dnn_layer/user_dnn_1/Relu:0
Buser_dnn_layer/user_dnn_layer/user_dnn_1/zero_fraction/cond/Cast:0
Puser_dnn_layer/user_dnn_layer/user_dnn_1/zero_fraction/cond/count_nonzero/Cast:0
Quser_dnn_layer/user_dnn_layer/user_dnn_1/zero_fraction/cond/count_nonzero/Const:0
[user_dnn_layer/user_dnn_layer/user_dnn_1/zero_fraction/cond/count_nonzero/NotEqual/Switch:1
Tuser_dnn_layer/user_dnn_layer/user_dnn_1/zero_fraction/cond/count_nonzero/NotEqual:0
Yuser_dnn_layer/user_dnn_layer/user_dnn_1/zero_fraction/cond/count_nonzero/nonzero_count:0
Quser_dnn_layer/user_dnn_layer/user_dnn_1/zero_fraction/cond/count_nonzero/zeros:0
Euser_dnn_layer/user_dnn_layer/user_dnn_1/zero_fraction/cond/pred_id:0
Fuser_dnn_layer/user_dnn_layer/user_dnn_1/zero_fraction/cond/switch_t:0
Euser_dnn_layer/user_dnn_layer/user_dnn_1/zero_fraction/cond/pred_id:0Euser_dnn_layer/user_dnn_layer/user_dnn_1/zero_fraction/cond/pred_id:0
/user_dnn_layer/user_dnn_layer/user_dnn_1/Relu:0[user_dnn_layer/user_dnn_layer/user_dnn_1/zero_fraction/cond/count_nonzero/NotEqual/Switch:1
Ń	
Guser_dnn_layer/user_dnn_layer/user_dnn_1/zero_fraction/cond/cond_text_1Euser_dnn_layer/user_dnn_layer/user_dnn_1/zero_fraction/cond/pred_id:0Fuser_dnn_layer/user_dnn_layer/user_dnn_1/zero_fraction/cond/switch_f:0*ö
/user_dnn_layer/user_dnn_layer/user_dnn_1/Relu:0
Ruser_dnn_layer/user_dnn_layer/user_dnn_1/zero_fraction/cond/count_nonzero_1/Cast:0
Suser_dnn_layer/user_dnn_layer/user_dnn_1/zero_fraction/cond/count_nonzero_1/Const:0
]user_dnn_layer/user_dnn_layer/user_dnn_1/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0
Vuser_dnn_layer/user_dnn_layer/user_dnn_1/zero_fraction/cond/count_nonzero_1/NotEqual:0
[user_dnn_layer/user_dnn_layer/user_dnn_1/zero_fraction/cond/count_nonzero_1/nonzero_count:0
Suser_dnn_layer/user_dnn_layer/user_dnn_1/zero_fraction/cond/count_nonzero_1/zeros:0
Euser_dnn_layer/user_dnn_layer/user_dnn_1/zero_fraction/cond/pred_id:0
Fuser_dnn_layer/user_dnn_layer/user_dnn_1/zero_fraction/cond/switch_f:0
/user_dnn_layer/user_dnn_layer/user_dnn_1/Relu:0]user_dnn_layer/user_dnn_layer/user_dnn_1/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0
Euser_dnn_layer/user_dnn_layer/user_dnn_1/zero_fraction/cond/pred_id:0Euser_dnn_layer/user_dnn_layer/user_dnn_1/zero_fraction/cond/pred_id:0


Euser_dnn_layer/user_dnn_layer/user_dnn_2/zero_fraction/cond/cond_textEuser_dnn_layer/user_dnn_layer/user_dnn_2/zero_fraction/cond/pred_id:0Fuser_dnn_layer/user_dnn_layer/user_dnn_2/zero_fraction/cond/switch_t:0 *Ź
/user_dnn_layer/user_dnn_layer/user_dnn_2/Relu:0
Buser_dnn_layer/user_dnn_layer/user_dnn_2/zero_fraction/cond/Cast:0
Puser_dnn_layer/user_dnn_layer/user_dnn_2/zero_fraction/cond/count_nonzero/Cast:0
Quser_dnn_layer/user_dnn_layer/user_dnn_2/zero_fraction/cond/count_nonzero/Const:0
[user_dnn_layer/user_dnn_layer/user_dnn_2/zero_fraction/cond/count_nonzero/NotEqual/Switch:1
Tuser_dnn_layer/user_dnn_layer/user_dnn_2/zero_fraction/cond/count_nonzero/NotEqual:0
Yuser_dnn_layer/user_dnn_layer/user_dnn_2/zero_fraction/cond/count_nonzero/nonzero_count:0
Quser_dnn_layer/user_dnn_layer/user_dnn_2/zero_fraction/cond/count_nonzero/zeros:0
Euser_dnn_layer/user_dnn_layer/user_dnn_2/zero_fraction/cond/pred_id:0
Fuser_dnn_layer/user_dnn_layer/user_dnn_2/zero_fraction/cond/switch_t:0
/user_dnn_layer/user_dnn_layer/user_dnn_2/Relu:0[user_dnn_layer/user_dnn_layer/user_dnn_2/zero_fraction/cond/count_nonzero/NotEqual/Switch:1
Euser_dnn_layer/user_dnn_layer/user_dnn_2/zero_fraction/cond/pred_id:0Euser_dnn_layer/user_dnn_layer/user_dnn_2/zero_fraction/cond/pred_id:0
Ń	
Guser_dnn_layer/user_dnn_layer/user_dnn_2/zero_fraction/cond/cond_text_1Euser_dnn_layer/user_dnn_layer/user_dnn_2/zero_fraction/cond/pred_id:0Fuser_dnn_layer/user_dnn_layer/user_dnn_2/zero_fraction/cond/switch_f:0*ö
/user_dnn_layer/user_dnn_layer/user_dnn_2/Relu:0
Ruser_dnn_layer/user_dnn_layer/user_dnn_2/zero_fraction/cond/count_nonzero_1/Cast:0
Suser_dnn_layer/user_dnn_layer/user_dnn_2/zero_fraction/cond/count_nonzero_1/Const:0
]user_dnn_layer/user_dnn_layer/user_dnn_2/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0
Vuser_dnn_layer/user_dnn_layer/user_dnn_2/zero_fraction/cond/count_nonzero_1/NotEqual:0
[user_dnn_layer/user_dnn_layer/user_dnn_2/zero_fraction/cond/count_nonzero_1/nonzero_count:0
Suser_dnn_layer/user_dnn_layer/user_dnn_2/zero_fraction/cond/count_nonzero_1/zeros:0
Euser_dnn_layer/user_dnn_layer/user_dnn_2/zero_fraction/cond/pred_id:0
Fuser_dnn_layer/user_dnn_layer/user_dnn_2/zero_fraction/cond/switch_f:0
Euser_dnn_layer/user_dnn_layer/user_dnn_2/zero_fraction/cond/pred_id:0Euser_dnn_layer/user_dnn_layer/user_dnn_2/zero_fraction/cond/pred_id:0
/user_dnn_layer/user_dnn_layer/user_dnn_2/Relu:0]user_dnn_layer/user_dnn_layer/user_dnn_2/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0


Euser_dnn_layer/user_dnn_layer/user_dnn_3/zero_fraction/cond/cond_textEuser_dnn_layer/user_dnn_layer/user_dnn_3/zero_fraction/cond/pred_id:0Fuser_dnn_layer/user_dnn_layer/user_dnn_3/zero_fraction/cond/switch_t:0 *ž
8user_dnn_layer/user_dnn_layer/user_dnn_3/dense/BiasAdd:0
Buser_dnn_layer/user_dnn_layer/user_dnn_3/zero_fraction/cond/Cast:0
Puser_dnn_layer/user_dnn_layer/user_dnn_3/zero_fraction/cond/count_nonzero/Cast:0
Quser_dnn_layer/user_dnn_layer/user_dnn_3/zero_fraction/cond/count_nonzero/Const:0
[user_dnn_layer/user_dnn_layer/user_dnn_3/zero_fraction/cond/count_nonzero/NotEqual/Switch:1
Tuser_dnn_layer/user_dnn_layer/user_dnn_3/zero_fraction/cond/count_nonzero/NotEqual:0
Yuser_dnn_layer/user_dnn_layer/user_dnn_3/zero_fraction/cond/count_nonzero/nonzero_count:0
Quser_dnn_layer/user_dnn_layer/user_dnn_3/zero_fraction/cond/count_nonzero/zeros:0
Euser_dnn_layer/user_dnn_layer/user_dnn_3/zero_fraction/cond/pred_id:0
Fuser_dnn_layer/user_dnn_layer/user_dnn_3/zero_fraction/cond/switch_t:0
8user_dnn_layer/user_dnn_layer/user_dnn_3/dense/BiasAdd:0[user_dnn_layer/user_dnn_layer/user_dnn_3/zero_fraction/cond/count_nonzero/NotEqual/Switch:1
Euser_dnn_layer/user_dnn_layer/user_dnn_3/zero_fraction/cond/pred_id:0Euser_dnn_layer/user_dnn_layer/user_dnn_3/zero_fraction/cond/pred_id:0
ă	
Guser_dnn_layer/user_dnn_layer/user_dnn_3/zero_fraction/cond/cond_text_1Euser_dnn_layer/user_dnn_layer/user_dnn_3/zero_fraction/cond/pred_id:0Fuser_dnn_layer/user_dnn_layer/user_dnn_3/zero_fraction/cond/switch_f:0*
8user_dnn_layer/user_dnn_layer/user_dnn_3/dense/BiasAdd:0
Ruser_dnn_layer/user_dnn_layer/user_dnn_3/zero_fraction/cond/count_nonzero_1/Cast:0
Suser_dnn_layer/user_dnn_layer/user_dnn_3/zero_fraction/cond/count_nonzero_1/Const:0
]user_dnn_layer/user_dnn_layer/user_dnn_3/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0
Vuser_dnn_layer/user_dnn_layer/user_dnn_3/zero_fraction/cond/count_nonzero_1/NotEqual:0
[user_dnn_layer/user_dnn_layer/user_dnn_3/zero_fraction/cond/count_nonzero_1/nonzero_count:0
Suser_dnn_layer/user_dnn_layer/user_dnn_3/zero_fraction/cond/count_nonzero_1/zeros:0
Euser_dnn_layer/user_dnn_layer/user_dnn_3/zero_fraction/cond/pred_id:0
Fuser_dnn_layer/user_dnn_layer/user_dnn_3/zero_fraction/cond/switch_f:0
Euser_dnn_layer/user_dnn_layer/user_dnn_3/zero_fraction/cond/pred_id:0Euser_dnn_layer/user_dnn_layer/user_dnn_3/zero_fraction/cond/pred_id:0
8user_dnn_layer/user_dnn_layer/user_dnn_3/dense/BiasAdd:0]user_dnn_layer/user_dnn_layer/user_dnn_3/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0


Eitem_dnn_layer/item_dnn_layer/item_dnn_0/zero_fraction/cond/cond_textEitem_dnn_layer/item_dnn_layer/item_dnn_0/zero_fraction/cond/pred_id:0Fitem_dnn_layer/item_dnn_layer/item_dnn_0/zero_fraction/cond/switch_t:0 *Ź
/item_dnn_layer/item_dnn_layer/item_dnn_0/Relu:0
Bitem_dnn_layer/item_dnn_layer/item_dnn_0/zero_fraction/cond/Cast:0
Pitem_dnn_layer/item_dnn_layer/item_dnn_0/zero_fraction/cond/count_nonzero/Cast:0
Qitem_dnn_layer/item_dnn_layer/item_dnn_0/zero_fraction/cond/count_nonzero/Const:0
[item_dnn_layer/item_dnn_layer/item_dnn_0/zero_fraction/cond/count_nonzero/NotEqual/Switch:1
Titem_dnn_layer/item_dnn_layer/item_dnn_0/zero_fraction/cond/count_nonzero/NotEqual:0
Yitem_dnn_layer/item_dnn_layer/item_dnn_0/zero_fraction/cond/count_nonzero/nonzero_count:0
Qitem_dnn_layer/item_dnn_layer/item_dnn_0/zero_fraction/cond/count_nonzero/zeros:0
Eitem_dnn_layer/item_dnn_layer/item_dnn_0/zero_fraction/cond/pred_id:0
Fitem_dnn_layer/item_dnn_layer/item_dnn_0/zero_fraction/cond/switch_t:0
Eitem_dnn_layer/item_dnn_layer/item_dnn_0/zero_fraction/cond/pred_id:0Eitem_dnn_layer/item_dnn_layer/item_dnn_0/zero_fraction/cond/pred_id:0
/item_dnn_layer/item_dnn_layer/item_dnn_0/Relu:0[item_dnn_layer/item_dnn_layer/item_dnn_0/zero_fraction/cond/count_nonzero/NotEqual/Switch:1
Ń	
Gitem_dnn_layer/item_dnn_layer/item_dnn_0/zero_fraction/cond/cond_text_1Eitem_dnn_layer/item_dnn_layer/item_dnn_0/zero_fraction/cond/pred_id:0Fitem_dnn_layer/item_dnn_layer/item_dnn_0/zero_fraction/cond/switch_f:0*ö
/item_dnn_layer/item_dnn_layer/item_dnn_0/Relu:0
Ritem_dnn_layer/item_dnn_layer/item_dnn_0/zero_fraction/cond/count_nonzero_1/Cast:0
Sitem_dnn_layer/item_dnn_layer/item_dnn_0/zero_fraction/cond/count_nonzero_1/Const:0
]item_dnn_layer/item_dnn_layer/item_dnn_0/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0
Vitem_dnn_layer/item_dnn_layer/item_dnn_0/zero_fraction/cond/count_nonzero_1/NotEqual:0
[item_dnn_layer/item_dnn_layer/item_dnn_0/zero_fraction/cond/count_nonzero_1/nonzero_count:0
Sitem_dnn_layer/item_dnn_layer/item_dnn_0/zero_fraction/cond/count_nonzero_1/zeros:0
Eitem_dnn_layer/item_dnn_layer/item_dnn_0/zero_fraction/cond/pred_id:0
Fitem_dnn_layer/item_dnn_layer/item_dnn_0/zero_fraction/cond/switch_f:0
Eitem_dnn_layer/item_dnn_layer/item_dnn_0/zero_fraction/cond/pred_id:0Eitem_dnn_layer/item_dnn_layer/item_dnn_0/zero_fraction/cond/pred_id:0
/item_dnn_layer/item_dnn_layer/item_dnn_0/Relu:0]item_dnn_layer/item_dnn_layer/item_dnn_0/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0


Eitem_dnn_layer/item_dnn_layer/item_dnn_1/zero_fraction/cond/cond_textEitem_dnn_layer/item_dnn_layer/item_dnn_1/zero_fraction/cond/pred_id:0Fitem_dnn_layer/item_dnn_layer/item_dnn_1/zero_fraction/cond/switch_t:0 *Ź
/item_dnn_layer/item_dnn_layer/item_dnn_1/Relu:0
Bitem_dnn_layer/item_dnn_layer/item_dnn_1/zero_fraction/cond/Cast:0
Pitem_dnn_layer/item_dnn_layer/item_dnn_1/zero_fraction/cond/count_nonzero/Cast:0
Qitem_dnn_layer/item_dnn_layer/item_dnn_1/zero_fraction/cond/count_nonzero/Const:0
[item_dnn_layer/item_dnn_layer/item_dnn_1/zero_fraction/cond/count_nonzero/NotEqual/Switch:1
Titem_dnn_layer/item_dnn_layer/item_dnn_1/zero_fraction/cond/count_nonzero/NotEqual:0
Yitem_dnn_layer/item_dnn_layer/item_dnn_1/zero_fraction/cond/count_nonzero/nonzero_count:0
Qitem_dnn_layer/item_dnn_layer/item_dnn_1/zero_fraction/cond/count_nonzero/zeros:0
Eitem_dnn_layer/item_dnn_layer/item_dnn_1/zero_fraction/cond/pred_id:0
Fitem_dnn_layer/item_dnn_layer/item_dnn_1/zero_fraction/cond/switch_t:0
Eitem_dnn_layer/item_dnn_layer/item_dnn_1/zero_fraction/cond/pred_id:0Eitem_dnn_layer/item_dnn_layer/item_dnn_1/zero_fraction/cond/pred_id:0
/item_dnn_layer/item_dnn_layer/item_dnn_1/Relu:0[item_dnn_layer/item_dnn_layer/item_dnn_1/zero_fraction/cond/count_nonzero/NotEqual/Switch:1
Ń	
Gitem_dnn_layer/item_dnn_layer/item_dnn_1/zero_fraction/cond/cond_text_1Eitem_dnn_layer/item_dnn_layer/item_dnn_1/zero_fraction/cond/pred_id:0Fitem_dnn_layer/item_dnn_layer/item_dnn_1/zero_fraction/cond/switch_f:0*ö
/item_dnn_layer/item_dnn_layer/item_dnn_1/Relu:0
Ritem_dnn_layer/item_dnn_layer/item_dnn_1/zero_fraction/cond/count_nonzero_1/Cast:0
Sitem_dnn_layer/item_dnn_layer/item_dnn_1/zero_fraction/cond/count_nonzero_1/Const:0
]item_dnn_layer/item_dnn_layer/item_dnn_1/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0
Vitem_dnn_layer/item_dnn_layer/item_dnn_1/zero_fraction/cond/count_nonzero_1/NotEqual:0
[item_dnn_layer/item_dnn_layer/item_dnn_1/zero_fraction/cond/count_nonzero_1/nonzero_count:0
Sitem_dnn_layer/item_dnn_layer/item_dnn_1/zero_fraction/cond/count_nonzero_1/zeros:0
Eitem_dnn_layer/item_dnn_layer/item_dnn_1/zero_fraction/cond/pred_id:0
Fitem_dnn_layer/item_dnn_layer/item_dnn_1/zero_fraction/cond/switch_f:0
Eitem_dnn_layer/item_dnn_layer/item_dnn_1/zero_fraction/cond/pred_id:0Eitem_dnn_layer/item_dnn_layer/item_dnn_1/zero_fraction/cond/pred_id:0
/item_dnn_layer/item_dnn_layer/item_dnn_1/Relu:0]item_dnn_layer/item_dnn_layer/item_dnn_1/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0


Eitem_dnn_layer/item_dnn_layer/item_dnn_2/zero_fraction/cond/cond_textEitem_dnn_layer/item_dnn_layer/item_dnn_2/zero_fraction/cond/pred_id:0Fitem_dnn_layer/item_dnn_layer/item_dnn_2/zero_fraction/cond/switch_t:0 *Ź
/item_dnn_layer/item_dnn_layer/item_dnn_2/Relu:0
Bitem_dnn_layer/item_dnn_layer/item_dnn_2/zero_fraction/cond/Cast:0
Pitem_dnn_layer/item_dnn_layer/item_dnn_2/zero_fraction/cond/count_nonzero/Cast:0
Qitem_dnn_layer/item_dnn_layer/item_dnn_2/zero_fraction/cond/count_nonzero/Const:0
[item_dnn_layer/item_dnn_layer/item_dnn_2/zero_fraction/cond/count_nonzero/NotEqual/Switch:1
Titem_dnn_layer/item_dnn_layer/item_dnn_2/zero_fraction/cond/count_nonzero/NotEqual:0
Yitem_dnn_layer/item_dnn_layer/item_dnn_2/zero_fraction/cond/count_nonzero/nonzero_count:0
Qitem_dnn_layer/item_dnn_layer/item_dnn_2/zero_fraction/cond/count_nonzero/zeros:0
Eitem_dnn_layer/item_dnn_layer/item_dnn_2/zero_fraction/cond/pred_id:0
Fitem_dnn_layer/item_dnn_layer/item_dnn_2/zero_fraction/cond/switch_t:0
Eitem_dnn_layer/item_dnn_layer/item_dnn_2/zero_fraction/cond/pred_id:0Eitem_dnn_layer/item_dnn_layer/item_dnn_2/zero_fraction/cond/pred_id:0
/item_dnn_layer/item_dnn_layer/item_dnn_2/Relu:0[item_dnn_layer/item_dnn_layer/item_dnn_2/zero_fraction/cond/count_nonzero/NotEqual/Switch:1
Ń	
Gitem_dnn_layer/item_dnn_layer/item_dnn_2/zero_fraction/cond/cond_text_1Eitem_dnn_layer/item_dnn_layer/item_dnn_2/zero_fraction/cond/pred_id:0Fitem_dnn_layer/item_dnn_layer/item_dnn_2/zero_fraction/cond/switch_f:0*ö
/item_dnn_layer/item_dnn_layer/item_dnn_2/Relu:0
Ritem_dnn_layer/item_dnn_layer/item_dnn_2/zero_fraction/cond/count_nonzero_1/Cast:0
Sitem_dnn_layer/item_dnn_layer/item_dnn_2/zero_fraction/cond/count_nonzero_1/Const:0
]item_dnn_layer/item_dnn_layer/item_dnn_2/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0
Vitem_dnn_layer/item_dnn_layer/item_dnn_2/zero_fraction/cond/count_nonzero_1/NotEqual:0
[item_dnn_layer/item_dnn_layer/item_dnn_2/zero_fraction/cond/count_nonzero_1/nonzero_count:0
Sitem_dnn_layer/item_dnn_layer/item_dnn_2/zero_fraction/cond/count_nonzero_1/zeros:0
Eitem_dnn_layer/item_dnn_layer/item_dnn_2/zero_fraction/cond/pred_id:0
Fitem_dnn_layer/item_dnn_layer/item_dnn_2/zero_fraction/cond/switch_f:0
Eitem_dnn_layer/item_dnn_layer/item_dnn_2/zero_fraction/cond/pred_id:0Eitem_dnn_layer/item_dnn_layer/item_dnn_2/zero_fraction/cond/pred_id:0
/item_dnn_layer/item_dnn_layer/item_dnn_2/Relu:0]item_dnn_layer/item_dnn_layer/item_dnn_2/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0


Eitem_dnn_layer/item_dnn_layer/item_dnn_3/zero_fraction/cond/cond_textEitem_dnn_layer/item_dnn_layer/item_dnn_3/zero_fraction/cond/pred_id:0Fitem_dnn_layer/item_dnn_layer/item_dnn_3/zero_fraction/cond/switch_t:0 *ž
8item_dnn_layer/item_dnn_layer/item_dnn_3/dense/BiasAdd:0
Bitem_dnn_layer/item_dnn_layer/item_dnn_3/zero_fraction/cond/Cast:0
Pitem_dnn_layer/item_dnn_layer/item_dnn_3/zero_fraction/cond/count_nonzero/Cast:0
Qitem_dnn_layer/item_dnn_layer/item_dnn_3/zero_fraction/cond/count_nonzero/Const:0
[item_dnn_layer/item_dnn_layer/item_dnn_3/zero_fraction/cond/count_nonzero/NotEqual/Switch:1
Titem_dnn_layer/item_dnn_layer/item_dnn_3/zero_fraction/cond/count_nonzero/NotEqual:0
Yitem_dnn_layer/item_dnn_layer/item_dnn_3/zero_fraction/cond/count_nonzero/nonzero_count:0
Qitem_dnn_layer/item_dnn_layer/item_dnn_3/zero_fraction/cond/count_nonzero/zeros:0
Eitem_dnn_layer/item_dnn_layer/item_dnn_3/zero_fraction/cond/pred_id:0
Fitem_dnn_layer/item_dnn_layer/item_dnn_3/zero_fraction/cond/switch_t:0
Eitem_dnn_layer/item_dnn_layer/item_dnn_3/zero_fraction/cond/pred_id:0Eitem_dnn_layer/item_dnn_layer/item_dnn_3/zero_fraction/cond/pred_id:0
8item_dnn_layer/item_dnn_layer/item_dnn_3/dense/BiasAdd:0[item_dnn_layer/item_dnn_layer/item_dnn_3/zero_fraction/cond/count_nonzero/NotEqual/Switch:1
ă	
Gitem_dnn_layer/item_dnn_layer/item_dnn_3/zero_fraction/cond/cond_text_1Eitem_dnn_layer/item_dnn_layer/item_dnn_3/zero_fraction/cond/pred_id:0Fitem_dnn_layer/item_dnn_layer/item_dnn_3/zero_fraction/cond/switch_f:0*
8item_dnn_layer/item_dnn_layer/item_dnn_3/dense/BiasAdd:0
Ritem_dnn_layer/item_dnn_layer/item_dnn_3/zero_fraction/cond/count_nonzero_1/Cast:0
Sitem_dnn_layer/item_dnn_layer/item_dnn_3/zero_fraction/cond/count_nonzero_1/Const:0
]item_dnn_layer/item_dnn_layer/item_dnn_3/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0
Vitem_dnn_layer/item_dnn_layer/item_dnn_3/zero_fraction/cond/count_nonzero_1/NotEqual:0
[item_dnn_layer/item_dnn_layer/item_dnn_3/zero_fraction/cond/count_nonzero_1/nonzero_count:0
Sitem_dnn_layer/item_dnn_layer/item_dnn_3/zero_fraction/cond/count_nonzero_1/zeros:0
Eitem_dnn_layer/item_dnn_layer/item_dnn_3/zero_fraction/cond/pred_id:0
Fitem_dnn_layer/item_dnn_layer/item_dnn_3/zero_fraction/cond/switch_f:0
Eitem_dnn_layer/item_dnn_layer/item_dnn_3/zero_fraction/cond/pred_id:0Eitem_dnn_layer/item_dnn_layer/item_dnn_3/zero_fraction/cond/pred_id:0
8item_dnn_layer/item_dnn_layer/item_dnn_3/dense/BiasAdd:0]item_dnn_layer/item_dnn_layer/item_dnn_3/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0"%
saved_model_main_op


group_deps"íŞ
	variablesŢŞÚŞ
Z
global_step:0global_step/Assignglobal_step/read:02global_step/Initializer/zeros:0H

Finput_layer/input_layer/age_level_embedding/embedding_weights/part_0:0Kinput_layer/input_layer/age_level_embedding/embedding_weights/part_0/AssignKinput_layer/input_layer/age_level_embedding/embedding_weights/part_0/read:0"K
=input_layer/input_layer/age_level_embedding/embedding_weights
  "
2cinput_layer/input_layer/age_level_embedding/embedding_weights/part_0/Initializer/truncated_normal:08
Ľ
Iinput_layer/input_layer/cms_group_id_embedding/embedding_weights/part_0:0Ninput_layer/input_layer/cms_group_id_embedding/embedding_weights/part_0/AssignNinput_layer/input_layer/cms_group_id_embedding/embedding_weights/part_0/read:0"N
@input_layer/input_layer/cms_group_id_embedding/embedding_weightsd  "d2finput_layer/input_layer/cms_group_id_embedding/embedding_weights/part_0/Initializer/truncated_normal:08

Finput_layer/input_layer/cms_segid_embedding/embedding_weights/part_0:0Kinput_layer/input_layer/cms_segid_embedding/embedding_weights/part_0/AssignKinput_layer/input_layer/cms_segid_embedding/embedding_weights/part_0/read:0"K
=input_layer/input_layer/cms_segid_embedding/embedding_weightsd  "d2cinput_layer/input_layer/cms_segid_embedding/embedding_weights/part_0/Initializer/truncated_normal:08
Í
Qinput_layer/input_layer/new_user_class_level_embedding/embedding_weights/part_0:0Vinput_layer/input_layer/new_user_class_level_embedding/embedding_weights/part_0/AssignVinput_layer/input_layer/new_user_class_level_embedding/embedding_weights/part_0/read:0"V
Hinput_layer/input_layer/new_user_class_level_embedding/embedding_weights
  "
2ninput_layer/input_layer/new_user_class_level_embedding/embedding_weights/part_0/Initializer/truncated_normal:08

Ginput_layer/input_layer/occupation_embedding/embedding_weights/part_0:0Linput_layer/input_layer/occupation_embedding/embedding_weights/part_0/AssignLinput_layer/input_layer/occupation_embedding/embedding_weights/part_0/read:0"L
>input_layer/input_layer/occupation_embedding/embedding_weights
  "
2dinput_layer/input_layer/occupation_embedding/embedding_weights/part_0/Initializer/truncated_normal:08
Ľ
Iinput_layer/input_layer/pvalue_level_embedding/embedding_weights/part_0:0Ninput_layer/input_layer/pvalue_level_embedding/embedding_weights/part_0/AssignNinput_layer/input_layer/pvalue_level_embedding/embedding_weights/part_0/read:0"N
@input_layer/input_layer/pvalue_level_embedding/embedding_weights
  "
2finput_layer/input_layer/pvalue_level_embedding/embedding_weights/part_0/Initializer/truncated_normal:08
Ż
Kinput_layer/input_layer/shopping_level_embedding/embedding_weights/part_0:0Pinput_layer/input_layer/shopping_level_embedding/embedding_weights/part_0/AssignPinput_layer/input_layer/shopping_level_embedding/embedding_weights/part_0/read:0"P
Binput_layer/input_layer/shopping_level_embedding/embedding_weights
  "
2hinput_layer/input_layer/shopping_level_embedding/embedding_weights/part_0/Initializer/truncated_normal:08

Dinput_layer/input_layer/user_id_embedding/embedding_weights/part_0:0Iinput_layer/input_layer/user_id_embedding/embedding_weights/part_0/AssignIinput_layer/input_layer/user_id_embedding/embedding_weights/part_0/read:0"M
;input_layer/input_layer/user_id_embedding/embedding_weights   " 2ainput_layer/input_layer/user_id_embedding/embedding_weights/part_0/Initializer/truncated_normal:08
Š
Iinput_layer/input_layer_1/adgroup_id_embedding/embedding_weights/part_0:0Ninput_layer/input_layer_1/adgroup_id_embedding/embedding_weights/part_0/AssignNinput_layer/input_layer_1/adgroup_id_embedding/embedding_weights/part_0/read:0"R
@input_layer/input_layer_1/adgroup_id_embedding/embedding_weights   " 2finput_layer/input_layer_1/adgroup_id_embedding/embedding_weights/part_0/Initializer/truncated_normal:08

Dinput_layer/input_layer_1/brand_embedding/embedding_weights/part_0:0Iinput_layer/input_layer_1/brand_embedding/embedding_weights/part_0/AssignIinput_layer/input_layer_1/brand_embedding/embedding_weights/part_0/read:0"M
;input_layer/input_layer_1/brand_embedding/embedding_weights   " 2ainput_layer/input_layer_1/brand_embedding/embedding_weights/part_0/Initializer/truncated_normal:08
Ž
Jinput_layer/input_layer_1/campaign_id_embedding/embedding_weights/part_0:0Oinput_layer/input_layer_1/campaign_id_embedding/embedding_weights/part_0/AssignOinput_layer/input_layer_1/campaign_id_embedding/embedding_weights/part_0/read:0"S
Ainput_layer/input_layer_1/campaign_id_embedding/embedding_weights   " 2ginput_layer/input_layer_1/campaign_id_embedding/embedding_weights/part_0/Initializer/truncated_normal:08

Finput_layer/input_layer_1/cate_id_embedding/embedding_weights/part_0:0Kinput_layer/input_layer_1/cate_id_embedding/embedding_weights/part_0/AssignKinput_layer/input_layer_1/cate_id_embedding/embedding_weights/part_0/read:0"M
=input_layer/input_layer_1/cate_id_embedding/embedding_weightsN  "N2cinput_layer/input_layer_1/cate_id_embedding/embedding_weights/part_0/Initializer/truncated_normal:08

Ginput_layer/input_layer_1/customer_embedding/embedding_weights/part_0:0Linput_layer/input_layer_1/customer_embedding/embedding_weights/part_0/AssignLinput_layer/input_layer_1/customer_embedding/embedding_weights/part_0/read:0"P
>input_layer/input_layer_1/customer_embedding/embedding_weights   " 2dinput_layer/input_layer_1/customer_embedding/embedding_weights/part_0/Initializer/truncated_normal:08
Č
Pinput_layer/input_layer_1/final_gender_code_embedding/embedding_weights/part_0:0Uinput_layer/input_layer_1/final_gender_code_embedding/embedding_weights/part_0/AssignUinput_layer/input_layer_1/final_gender_code_embedding/embedding_weights/part_0/read:0"U
Ginput_layer/input_layer_1/final_gender_code_embedding/embedding_weights
  "
2minput_layer/input_layer_1/final_gender_code_embedding/embedding_weights/part_0/Initializer/truncated_normal:08

Binput_layer/input_layer_1/pid_embedding/embedding_weights/part_0:0Ginput_layer/input_layer_1/pid_embedding/embedding_weights/part_0/AssignGinput_layer/input_layer_1/pid_embedding/embedding_weights/part_0/read:0"G
9input_layer/input_layer_1/pid_embedding/embedding_weights
  "
2_input_layer/input_layer_1/pid_embedding/embedding_weights/part_0/Initializer/truncated_normal:08

Dinput_layer/input_layer_1/price_embedding/embedding_weights/part_0:0Iinput_layer/input_layer_1/price_embedding/embedding_weights/part_0/AssignIinput_layer/input_layer_1/price_embedding/embedding_weights/part_0/read:0"I
;input_layer/input_layer_1/price_embedding/embedding_weights2  "22ainput_layer/input_layer_1/price_embedding/embedding_weights/part_0/Initializer/truncated_normal:08
đ
>user_dnn_layer/user_dnn_layer/user_dnn_0/dense/kernel/part_0:0Cuser_dnn_layer/user_dnn_layer/user_dnn_0/dense/kernel/part_0/AssignCuser_dnn_layer/user_dnn_layer/user_dnn_0/dense/kernel/part_0/read:0"G
5user_dnn_layer/user_dnn_layer/user_dnn_0/dense/kernel  "2Yuser_dnn_layer/user_dnn_layer/user_dnn_0/dense/kernel/part_0/Initializer/random_uniform:08
Ř
<user_dnn_layer/user_dnn_layer/user_dnn_0/dense/bias/part_0:0Auser_dnn_layer/user_dnn_layer/user_dnn_0/dense/bias/part_0/AssignAuser_dnn_layer/user_dnn_layer/user_dnn_0/dense/bias/part_0/read:0"@
3user_dnn_layer/user_dnn_layer/user_dnn_0/dense/bias "2Nuser_dnn_layer/user_dnn_layer/user_dnn_0/dense/bias/part_0/Initializer/zeros:08
˘
Kuser_dnn_layer/user_dnn_layer/user_dnn_0/batch_normalization/gamma/part_0:0Puser_dnn_layer/user_dnn_layer/user_dnn_0/batch_normalization/gamma/part_0/AssignPuser_dnn_layer/user_dnn_layer/user_dnn_0/batch_normalization/gamma/part_0/read:0"O
Buser_dnn_layer/user_dnn_layer/user_dnn_0/batch_normalization/gamma "2\user_dnn_layer/user_dnn_layer/user_dnn_0/batch_normalization/gamma/part_0/Initializer/ones:08

Juser_dnn_layer/user_dnn_layer/user_dnn_0/batch_normalization/beta/part_0:0Ouser_dnn_layer/user_dnn_layer/user_dnn_0/batch_normalization/beta/part_0/AssignOuser_dnn_layer/user_dnn_layer/user_dnn_0/batch_normalization/beta/part_0/read:0"N
Auser_dnn_layer/user_dnn_layer/user_dnn_0/batch_normalization/beta "2\user_dnn_layer/user_dnn_layer/user_dnn_0/batch_normalization/beta/part_0/Initializer/zeros:08
Đ
Juser_dnn_layer/user_dnn_layer/user_dnn_0/batch_normalization/moving_mean:0Ouser_dnn_layer/user_dnn_layer/user_dnn_0/batch_normalization/moving_mean/AssignOuser_dnn_layer/user_dnn_layer/user_dnn_0/batch_normalization/moving_mean/read:02\user_dnn_layer/user_dnn_layer/user_dnn_0/batch_normalization/moving_mean/Initializer/zeros:0@H
ß
Nuser_dnn_layer/user_dnn_layer/user_dnn_0/batch_normalization/moving_variance:0Suser_dnn_layer/user_dnn_layer/user_dnn_0/batch_normalization/moving_variance/AssignSuser_dnn_layer/user_dnn_layer/user_dnn_0/batch_normalization/moving_variance/read:02_user_dnn_layer/user_dnn_layer/user_dnn_0/batch_normalization/moving_variance/Initializer/ones:0@H
đ
>user_dnn_layer/user_dnn_layer/user_dnn_1/dense/kernel/part_0:0Cuser_dnn_layer/user_dnn_layer/user_dnn_1/dense/kernel/part_0/AssignCuser_dnn_layer/user_dnn_layer/user_dnn_1/dense/kernel/part_0/read:0"G
5user_dnn_layer/user_dnn_layer/user_dnn_1/dense/kernel  "2Yuser_dnn_layer/user_dnn_layer/user_dnn_1/dense/kernel/part_0/Initializer/random_uniform:08
Ř
<user_dnn_layer/user_dnn_layer/user_dnn_1/dense/bias/part_0:0Auser_dnn_layer/user_dnn_layer/user_dnn_1/dense/bias/part_0/AssignAuser_dnn_layer/user_dnn_layer/user_dnn_1/dense/bias/part_0/read:0"@
3user_dnn_layer/user_dnn_layer/user_dnn_1/dense/bias "2Nuser_dnn_layer/user_dnn_layer/user_dnn_1/dense/bias/part_0/Initializer/zeros:08
˘
Kuser_dnn_layer/user_dnn_layer/user_dnn_1/batch_normalization/gamma/part_0:0Puser_dnn_layer/user_dnn_layer/user_dnn_1/batch_normalization/gamma/part_0/AssignPuser_dnn_layer/user_dnn_layer/user_dnn_1/batch_normalization/gamma/part_0/read:0"O
Buser_dnn_layer/user_dnn_layer/user_dnn_1/batch_normalization/gamma "2\user_dnn_layer/user_dnn_layer/user_dnn_1/batch_normalization/gamma/part_0/Initializer/ones:08

Juser_dnn_layer/user_dnn_layer/user_dnn_1/batch_normalization/beta/part_0:0Ouser_dnn_layer/user_dnn_layer/user_dnn_1/batch_normalization/beta/part_0/AssignOuser_dnn_layer/user_dnn_layer/user_dnn_1/batch_normalization/beta/part_0/read:0"N
Auser_dnn_layer/user_dnn_layer/user_dnn_1/batch_normalization/beta "2\user_dnn_layer/user_dnn_layer/user_dnn_1/batch_normalization/beta/part_0/Initializer/zeros:08
Đ
Juser_dnn_layer/user_dnn_layer/user_dnn_1/batch_normalization/moving_mean:0Ouser_dnn_layer/user_dnn_layer/user_dnn_1/batch_normalization/moving_mean/AssignOuser_dnn_layer/user_dnn_layer/user_dnn_1/batch_normalization/moving_mean/read:02\user_dnn_layer/user_dnn_layer/user_dnn_1/batch_normalization/moving_mean/Initializer/zeros:0@H
ß
Nuser_dnn_layer/user_dnn_layer/user_dnn_1/batch_normalization/moving_variance:0Suser_dnn_layer/user_dnn_layer/user_dnn_1/batch_normalization/moving_variance/AssignSuser_dnn_layer/user_dnn_layer/user_dnn_1/batch_normalization/moving_variance/read:02_user_dnn_layer/user_dnn_layer/user_dnn_1/batch_normalization/moving_variance/Initializer/ones:0@H
î
>user_dnn_layer/user_dnn_layer/user_dnn_2/dense/kernel/part_0:0Cuser_dnn_layer/user_dnn_layer/user_dnn_2/dense/kernel/part_0/AssignCuser_dnn_layer/user_dnn_layer/user_dnn_2/dense/kernel/part_0/read:0"E
5user_dnn_layer/user_dnn_layer/user_dnn_2/dense/kernel@  "@2Yuser_dnn_layer/user_dnn_layer/user_dnn_2/dense/kernel/part_0/Initializer/random_uniform:08
Ö
<user_dnn_layer/user_dnn_layer/user_dnn_2/dense/bias/part_0:0Auser_dnn_layer/user_dnn_layer/user_dnn_2/dense/bias/part_0/AssignAuser_dnn_layer/user_dnn_layer/user_dnn_2/dense/bias/part_0/read:0">
3user_dnn_layer/user_dnn_layer/user_dnn_2/dense/bias@ "@2Nuser_dnn_layer/user_dnn_layer/user_dnn_2/dense/bias/part_0/Initializer/zeros:08
 
Kuser_dnn_layer/user_dnn_layer/user_dnn_2/batch_normalization/gamma/part_0:0Puser_dnn_layer/user_dnn_layer/user_dnn_2/batch_normalization/gamma/part_0/AssignPuser_dnn_layer/user_dnn_layer/user_dnn_2/batch_normalization/gamma/part_0/read:0"M
Buser_dnn_layer/user_dnn_layer/user_dnn_2/batch_normalization/gamma@ "@2\user_dnn_layer/user_dnn_layer/user_dnn_2/batch_normalization/gamma/part_0/Initializer/ones:08

Juser_dnn_layer/user_dnn_layer/user_dnn_2/batch_normalization/beta/part_0:0Ouser_dnn_layer/user_dnn_layer/user_dnn_2/batch_normalization/beta/part_0/AssignOuser_dnn_layer/user_dnn_layer/user_dnn_2/batch_normalization/beta/part_0/read:0"L
Auser_dnn_layer/user_dnn_layer/user_dnn_2/batch_normalization/beta@ "@2\user_dnn_layer/user_dnn_layer/user_dnn_2/batch_normalization/beta/part_0/Initializer/zeros:08
Đ
Juser_dnn_layer/user_dnn_layer/user_dnn_2/batch_normalization/moving_mean:0Ouser_dnn_layer/user_dnn_layer/user_dnn_2/batch_normalization/moving_mean/AssignOuser_dnn_layer/user_dnn_layer/user_dnn_2/batch_normalization/moving_mean/read:02\user_dnn_layer/user_dnn_layer/user_dnn_2/batch_normalization/moving_mean/Initializer/zeros:0@H
ß
Nuser_dnn_layer/user_dnn_layer/user_dnn_2/batch_normalization/moving_variance:0Suser_dnn_layer/user_dnn_layer/user_dnn_2/batch_normalization/moving_variance/AssignSuser_dnn_layer/user_dnn_layer/user_dnn_2/batch_normalization/moving_variance/read:02_user_dnn_layer/user_dnn_layer/user_dnn_2/batch_normalization/moving_variance/Initializer/ones:0@H
ě
>user_dnn_layer/user_dnn_layer/user_dnn_3/dense/kernel/part_0:0Cuser_dnn_layer/user_dnn_layer/user_dnn_3/dense/kernel/part_0/AssignCuser_dnn_layer/user_dnn_layer/user_dnn_3/dense/kernel/part_0/read:0"C
5user_dnn_layer/user_dnn_layer/user_dnn_3/dense/kernel@   "@ 2Yuser_dnn_layer/user_dnn_layer/user_dnn_3/dense/kernel/part_0/Initializer/random_uniform:08
Ö
<user_dnn_layer/user_dnn_layer/user_dnn_3/dense/bias/part_0:0Auser_dnn_layer/user_dnn_layer/user_dnn_3/dense/bias/part_0/AssignAuser_dnn_layer/user_dnn_layer/user_dnn_3/dense/bias/part_0/read:0">
3user_dnn_layer/user_dnn_layer/user_dnn_3/dense/bias  " 2Nuser_dnn_layer/user_dnn_layer/user_dnn_3/dense/bias/part_0/Initializer/zeros:08
đ
>item_dnn_layer/item_dnn_layer/item_dnn_0/dense/kernel/part_0:0Citem_dnn_layer/item_dnn_layer/item_dnn_0/dense/kernel/part_0/AssignCitem_dnn_layer/item_dnn_layer/item_dnn_0/dense/kernel/part_0/read:0"G
5item_dnn_layer/item_dnn_layer/item_dnn_0/dense/kernel  "2Yitem_dnn_layer/item_dnn_layer/item_dnn_0/dense/kernel/part_0/Initializer/random_uniform:08
Ř
<item_dnn_layer/item_dnn_layer/item_dnn_0/dense/bias/part_0:0Aitem_dnn_layer/item_dnn_layer/item_dnn_0/dense/bias/part_0/AssignAitem_dnn_layer/item_dnn_layer/item_dnn_0/dense/bias/part_0/read:0"@
3item_dnn_layer/item_dnn_layer/item_dnn_0/dense/bias "2Nitem_dnn_layer/item_dnn_layer/item_dnn_0/dense/bias/part_0/Initializer/zeros:08
˘
Kitem_dnn_layer/item_dnn_layer/item_dnn_0/batch_normalization/gamma/part_0:0Pitem_dnn_layer/item_dnn_layer/item_dnn_0/batch_normalization/gamma/part_0/AssignPitem_dnn_layer/item_dnn_layer/item_dnn_0/batch_normalization/gamma/part_0/read:0"O
Bitem_dnn_layer/item_dnn_layer/item_dnn_0/batch_normalization/gamma "2\item_dnn_layer/item_dnn_layer/item_dnn_0/batch_normalization/gamma/part_0/Initializer/ones:08

Jitem_dnn_layer/item_dnn_layer/item_dnn_0/batch_normalization/beta/part_0:0Oitem_dnn_layer/item_dnn_layer/item_dnn_0/batch_normalization/beta/part_0/AssignOitem_dnn_layer/item_dnn_layer/item_dnn_0/batch_normalization/beta/part_0/read:0"N
Aitem_dnn_layer/item_dnn_layer/item_dnn_0/batch_normalization/beta "2\item_dnn_layer/item_dnn_layer/item_dnn_0/batch_normalization/beta/part_0/Initializer/zeros:08
Đ
Jitem_dnn_layer/item_dnn_layer/item_dnn_0/batch_normalization/moving_mean:0Oitem_dnn_layer/item_dnn_layer/item_dnn_0/batch_normalization/moving_mean/AssignOitem_dnn_layer/item_dnn_layer/item_dnn_0/batch_normalization/moving_mean/read:02\item_dnn_layer/item_dnn_layer/item_dnn_0/batch_normalization/moving_mean/Initializer/zeros:0@H
ß
Nitem_dnn_layer/item_dnn_layer/item_dnn_0/batch_normalization/moving_variance:0Sitem_dnn_layer/item_dnn_layer/item_dnn_0/batch_normalization/moving_variance/AssignSitem_dnn_layer/item_dnn_layer/item_dnn_0/batch_normalization/moving_variance/read:02_item_dnn_layer/item_dnn_layer/item_dnn_0/batch_normalization/moving_variance/Initializer/ones:0@H
đ
>item_dnn_layer/item_dnn_layer/item_dnn_1/dense/kernel/part_0:0Citem_dnn_layer/item_dnn_layer/item_dnn_1/dense/kernel/part_0/AssignCitem_dnn_layer/item_dnn_layer/item_dnn_1/dense/kernel/part_0/read:0"G
5item_dnn_layer/item_dnn_layer/item_dnn_1/dense/kernel  "2Yitem_dnn_layer/item_dnn_layer/item_dnn_1/dense/kernel/part_0/Initializer/random_uniform:08
Ř
<item_dnn_layer/item_dnn_layer/item_dnn_1/dense/bias/part_0:0Aitem_dnn_layer/item_dnn_layer/item_dnn_1/dense/bias/part_0/AssignAitem_dnn_layer/item_dnn_layer/item_dnn_1/dense/bias/part_0/read:0"@
3item_dnn_layer/item_dnn_layer/item_dnn_1/dense/bias "2Nitem_dnn_layer/item_dnn_layer/item_dnn_1/dense/bias/part_0/Initializer/zeros:08
˘
Kitem_dnn_layer/item_dnn_layer/item_dnn_1/batch_normalization/gamma/part_0:0Pitem_dnn_layer/item_dnn_layer/item_dnn_1/batch_normalization/gamma/part_0/AssignPitem_dnn_layer/item_dnn_layer/item_dnn_1/batch_normalization/gamma/part_0/read:0"O
Bitem_dnn_layer/item_dnn_layer/item_dnn_1/batch_normalization/gamma "2\item_dnn_layer/item_dnn_layer/item_dnn_1/batch_normalization/gamma/part_0/Initializer/ones:08

Jitem_dnn_layer/item_dnn_layer/item_dnn_1/batch_normalization/beta/part_0:0Oitem_dnn_layer/item_dnn_layer/item_dnn_1/batch_normalization/beta/part_0/AssignOitem_dnn_layer/item_dnn_layer/item_dnn_1/batch_normalization/beta/part_0/read:0"N
Aitem_dnn_layer/item_dnn_layer/item_dnn_1/batch_normalization/beta "2\item_dnn_layer/item_dnn_layer/item_dnn_1/batch_normalization/beta/part_0/Initializer/zeros:08
Đ
Jitem_dnn_layer/item_dnn_layer/item_dnn_1/batch_normalization/moving_mean:0Oitem_dnn_layer/item_dnn_layer/item_dnn_1/batch_normalization/moving_mean/AssignOitem_dnn_layer/item_dnn_layer/item_dnn_1/batch_normalization/moving_mean/read:02\item_dnn_layer/item_dnn_layer/item_dnn_1/batch_normalization/moving_mean/Initializer/zeros:0@H
ß
Nitem_dnn_layer/item_dnn_layer/item_dnn_1/batch_normalization/moving_variance:0Sitem_dnn_layer/item_dnn_layer/item_dnn_1/batch_normalization/moving_variance/AssignSitem_dnn_layer/item_dnn_layer/item_dnn_1/batch_normalization/moving_variance/read:02_item_dnn_layer/item_dnn_layer/item_dnn_1/batch_normalization/moving_variance/Initializer/ones:0@H
î
>item_dnn_layer/item_dnn_layer/item_dnn_2/dense/kernel/part_0:0Citem_dnn_layer/item_dnn_layer/item_dnn_2/dense/kernel/part_0/AssignCitem_dnn_layer/item_dnn_layer/item_dnn_2/dense/kernel/part_0/read:0"E
5item_dnn_layer/item_dnn_layer/item_dnn_2/dense/kernel@  "@2Yitem_dnn_layer/item_dnn_layer/item_dnn_2/dense/kernel/part_0/Initializer/random_uniform:08
Ö
<item_dnn_layer/item_dnn_layer/item_dnn_2/dense/bias/part_0:0Aitem_dnn_layer/item_dnn_layer/item_dnn_2/dense/bias/part_0/AssignAitem_dnn_layer/item_dnn_layer/item_dnn_2/dense/bias/part_0/read:0">
3item_dnn_layer/item_dnn_layer/item_dnn_2/dense/bias@ "@2Nitem_dnn_layer/item_dnn_layer/item_dnn_2/dense/bias/part_0/Initializer/zeros:08
 
Kitem_dnn_layer/item_dnn_layer/item_dnn_2/batch_normalization/gamma/part_0:0Pitem_dnn_layer/item_dnn_layer/item_dnn_2/batch_normalization/gamma/part_0/AssignPitem_dnn_layer/item_dnn_layer/item_dnn_2/batch_normalization/gamma/part_0/read:0"M
Bitem_dnn_layer/item_dnn_layer/item_dnn_2/batch_normalization/gamma@ "@2\item_dnn_layer/item_dnn_layer/item_dnn_2/batch_normalization/gamma/part_0/Initializer/ones:08

Jitem_dnn_layer/item_dnn_layer/item_dnn_2/batch_normalization/beta/part_0:0Oitem_dnn_layer/item_dnn_layer/item_dnn_2/batch_normalization/beta/part_0/AssignOitem_dnn_layer/item_dnn_layer/item_dnn_2/batch_normalization/beta/part_0/read:0"L
Aitem_dnn_layer/item_dnn_layer/item_dnn_2/batch_normalization/beta@ "@2\item_dnn_layer/item_dnn_layer/item_dnn_2/batch_normalization/beta/part_0/Initializer/zeros:08
Đ
Jitem_dnn_layer/item_dnn_layer/item_dnn_2/batch_normalization/moving_mean:0Oitem_dnn_layer/item_dnn_layer/item_dnn_2/batch_normalization/moving_mean/AssignOitem_dnn_layer/item_dnn_layer/item_dnn_2/batch_normalization/moving_mean/read:02\item_dnn_layer/item_dnn_layer/item_dnn_2/batch_normalization/moving_mean/Initializer/zeros:0@H
ß
Nitem_dnn_layer/item_dnn_layer/item_dnn_2/batch_normalization/moving_variance:0Sitem_dnn_layer/item_dnn_layer/item_dnn_2/batch_normalization/moving_variance/AssignSitem_dnn_layer/item_dnn_layer/item_dnn_2/batch_normalization/moving_variance/read:02_item_dnn_layer/item_dnn_layer/item_dnn_2/batch_normalization/moving_variance/Initializer/ones:0@H
ě
>item_dnn_layer/item_dnn_layer/item_dnn_3/dense/kernel/part_0:0Citem_dnn_layer/item_dnn_layer/item_dnn_3/dense/kernel/part_0/AssignCitem_dnn_layer/item_dnn_layer/item_dnn_3/dense/kernel/part_0/read:0"C
5item_dnn_layer/item_dnn_layer/item_dnn_3/dense/kernel@   "@ 2Yitem_dnn_layer/item_dnn_layer/item_dnn_3/dense/kernel/part_0/Initializer/random_uniform:08
Ö
<item_dnn_layer/item_dnn_layer/item_dnn_3/dense/bias/part_0:0Aitem_dnn_layer/item_dnn_layer/item_dnn_3/dense/bias/part_0/AssignAitem_dnn_layer/item_dnn_layer/item_dnn_3/dense/bias/part_0/read:0">
3item_dnn_layer/item_dnn_layer/item_dnn_3/dense/bias  " 2Nitem_dnn_layer/item_dnn_layer/item_dnn_3/dense/bias/part_0/Initializer/zeros:08
A
sim_w:0sim_w/Assignsim_w/read:02sim_w/Initializer/ones:08
B
sim_b:0sim_b/Assignsim_b/read:02sim_b/Initializer/zeros:08*Ň
serving_defaultž
'
clk 
Placeholder:0˙˙˙˙˙˙˙˙˙
/
	cms_segid"
Placeholder_9:0˙˙˙˙˙˙˙˙˙
8
final_gender_code#
Placeholder_11:0˙˙˙˙˙˙˙˙˙
0

adgroup_id"
Placeholder_3:0˙˙˙˙˙˙˙˙˙
3
pvalue_level#
Placeholder_13:0˙˙˙˙˙˙˙˙˙
)
buy"
Placeholder_1:0˙˙˙˙˙˙˙˙˙
-
user_id"
Placeholder_8:0˙˙˙˙˙˙˙˙˙
;
new_user_class_level#
Placeholder_16:0˙˙˙˙˙˙˙˙˙
.
customer"
Placeholder_6:0˙˙˙˙˙˙˙˙˙
-
cate_id"
Placeholder_4:0˙˙˙˙˙˙˙˙˙
3
cms_group_id#
Placeholder_10:0˙˙˙˙˙˙˙˙˙
,
price#
Placeholder_17:0˙˙˙˙˙˙˙˙˙
)
pid"
Placeholder_2:0˙˙˙˙˙˙˙˙˙
5
shopping_level#
Placeholder_14:0˙˙˙˙˙˙˙˙˙
1

occupation#
Placeholder_15:0˙˙˙˙˙˙˙˙˙
1
campaign_id"
Placeholder_5:0˙˙˙˙˙˙˙˙˙
0
	age_level#
Placeholder_12:0˙˙˙˙˙˙˙˙˙
+
brand"
Placeholder_7:0˙˙˙˙˙˙˙˙˙%
score
	Sigmoid:0˙˙˙˙˙˙˙˙˙tensorflow/serving/predict