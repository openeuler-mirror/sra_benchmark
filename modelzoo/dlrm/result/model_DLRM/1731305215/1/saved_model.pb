эИ4
ф*«*
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
2	АР
x
Assign
ref"TА

value"T

output_ref"TА"	
Ttype"
validate_shapebool("
use_lockingbool(Ш
s
	AssignSub
ref"TА

value"T

output_ref"TА" 
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
≠
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
Н
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
delete_old_dirsbool(И
=
Mul
x"T
y"T
z"T"
Ttype:
2	Р
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
incompatible_shape_errorbool(Р
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
Н
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
2	И
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
list(type)(0И
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
list(type)(0И
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
Ј
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
G
SquaredDifference
x"T
y"T
z"T"
Ttype:

2	Р
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
ц
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
М
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
А
TruncatedNormal

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	И
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
ref"dtypeА"
shapeshape"
dtypetype"
	containerstring "
shared_namestring И
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
Ttype"serve*1.15.02v1.15.0-rc3-22-g590d6ee8лс/

global_step/Initializer/zerosConst*
value	B	 R *
dtype0	*
_class
loc:@global_step*
_output_shapes
: 
k
global_step
VariableV2*
_output_shapes
: *
shape: *
_class
loc:@global_step*
dtype0	
Й
global_step/AssignAssignglobal_stepglobal_step/Initializer/zeros*
_output_shapes
: *
_class
loc:@global_step*
T0	
j
global_step/readIdentityglobal_step*
T0	*
_output_shapes
: *
_class
loc:@global_step
f
PlaceholderPlaceholder*#
_output_shapes
:€€€€€€€€€*
shape:€€€€€€€€€*
dtype0
h
Placeholder_1Placeholder*
dtype0*
shape:€€€€€€€€€*#
_output_shapes
:€€€€€€€€€
h
Placeholder_2Placeholder*
dtype0*
shape:€€€€€€€€€*#
_output_shapes
:€€€€€€€€€
h
Placeholder_3Placeholder*#
_output_shapes
:€€€€€€€€€*
shape:€€€€€€€€€*
dtype0
h
Placeholder_4Placeholder*
dtype0*#
_output_shapes
:€€€€€€€€€*
shape:€€€€€€€€€
h
Placeholder_5Placeholder*
dtype0*
shape:€€€€€€€€€*#
_output_shapes
:€€€€€€€€€
h
Placeholder_6Placeholder*
shape:€€€€€€€€€*
dtype0*#
_output_shapes
:€€€€€€€€€
h
Placeholder_7Placeholder*
dtype0*
shape:€€€€€€€€€*#
_output_shapes
:€€€€€€€€€
h
Placeholder_8Placeholder*#
_output_shapes
:€€€€€€€€€*
shape:€€€€€€€€€*
dtype0
h
Placeholder_9Placeholder*
dtype0*
shape:€€€€€€€€€*#
_output_shapes
:€€€€€€€€€
i
Placeholder_10Placeholder*
dtype0*
shape:€€€€€€€€€*#
_output_shapes
:€€€€€€€€€
i
Placeholder_11Placeholder*
dtype0*#
_output_shapes
:€€€€€€€€€*
shape:€€€€€€€€€
i
Placeholder_12Placeholder*
dtype0*#
_output_shapes
:€€€€€€€€€*
shape:€€€€€€€€€
i
Placeholder_13Placeholder*
dtype0*#
_output_shapes
:€€€€€€€€€*
shape:€€€€€€€€€
i
Placeholder_14Placeholder*#
_output_shapes
:€€€€€€€€€*
dtype0*
shape:€€€€€€€€€
i
Placeholder_15Placeholder*
shape:€€€€€€€€€*#
_output_shapes
:€€€€€€€€€*
dtype0
i
Placeholder_16Placeholder*#
_output_shapes
:€€€€€€€€€*
dtype0*
shape:€€€€€€€€€
i
Placeholder_17Placeholder*#
_output_shapes
:€€€€€€€€€*
dtype0*
shape:€€€€€€€€€
i
Placeholder_18Placeholder*
dtype0*#
_output_shapes
:€€€€€€€€€*
shape:€€€€€€€€€
i
Placeholder_19Placeholder*
shape:€€€€€€€€€*
dtype0*#
_output_shapes
:€€€€€€€€€
i
Placeholder_20Placeholder*
shape:€€€€€€€€€*
dtype0*#
_output_shapes
:€€€€€€€€€
i
Placeholder_21Placeholder*
shape:€€€€€€€€€*#
_output_shapes
:€€€€€€€€€*
dtype0
i
Placeholder_22Placeholder*
dtype0*
shape:€€€€€€€€€*#
_output_shapes
:€€€€€€€€€
i
Placeholder_23Placeholder*
shape:€€€€€€€€€*#
_output_shapes
:€€€€€€€€€*
dtype0
i
Placeholder_24Placeholder*#
_output_shapes
:€€€€€€€€€*
dtype0*
shape:€€€€€€€€€
i
Placeholder_25Placeholder*#
_output_shapes
:€€€€€€€€€*
dtype0*
shape:€€€€€€€€€
i
Placeholder_26Placeholder*#
_output_shapes
:€€€€€€€€€*
dtype0*
shape:€€€€€€€€€
i
Placeholder_27Placeholder*
dtype0*
shape:€€€€€€€€€*#
_output_shapes
:€€€€€€€€€
i
Placeholder_28Placeholder*
dtype0*
shape:€€€€€€€€€*#
_output_shapes
:€€€€€€€€€
i
Placeholder_29Placeholder*
shape:€€€€€€€€€*#
_output_shapes
:€€€€€€€€€*
dtype0
i
Placeholder_30Placeholder*
dtype0*#
_output_shapes
:€€€€€€€€€*
shape:€€€€€€€€€
i
Placeholder_31Placeholder*#
_output_shapes
:€€€€€€€€€*
dtype0*
shape:€€€€€€€€€
i
Placeholder_32Placeholder*
dtype0*
shape:€€€€€€€€€*#
_output_shapes
:€€€€€€€€€
i
Placeholder_33Placeholder*
shape:€€€€€€€€€*#
_output_shapes
:€€€€€€€€€*
dtype0
i
Placeholder_34Placeholder*#
_output_shapes
:€€€€€€€€€*
shape:€€€€€€€€€*
dtype0
i
Placeholder_35Placeholder*
shape:€€€€€€€€€*
dtype0*#
_output_shapes
:€€€€€€€€€
i
Placeholder_36Placeholder*
shape:€€€€€€€€€*#
_output_shapes
:€€€€€€€€€*
dtype0
i
Placeholder_37Placeholder*#
_output_shapes
:€€€€€€€€€*
shape:€€€€€€€€€*
dtype0
i
Placeholder_38Placeholder*#
_output_shapes
:€€€€€€€€€*
shape:€€€€€€€€€*
dtype0
Ж
;input_layer/dense_input_layer/input_layer/I1/ExpandDims/dimConst*
dtype0*
_output_shapes
: *
valueB :
€€€€€€€€€
Ѕ
7input_layer/dense_input_layer/input_layer/I1/ExpandDims
ExpandDimsPlaceholder;input_layer/dense_input_layer/input_layer/I1/ExpandDims/dim*
T0*'
_output_shapes
:€€€€€€€€€
Щ
2input_layer/dense_input_layer/input_layer/I1/ShapeShape7input_layer/dense_input_layer/input_layer/I1/ExpandDims*
T0*
_output_shapes
:
К
@input_layer/dense_input_layer/input_layer/I1/strided_slice/stackConst*
dtype0*
valueB: *
_output_shapes
:
М
Binput_layer/dense_input_layer/input_layer/I1/strided_slice/stack_1Const*
dtype0*
valueB:*
_output_shapes
:
М
Binput_layer/dense_input_layer/input_layer/I1/strided_slice/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
О
:input_layer/dense_input_layer/input_layer/I1/strided_sliceStridedSlice2input_layer/dense_input_layer/input_layer/I1/Shape@input_layer/dense_input_layer/input_layer/I1/strided_slice/stackBinput_layer/dense_input_layer/input_layer/I1/strided_slice/stack_1Binput_layer/dense_input_layer/input_layer/I1/strided_slice/stack_2*
Index0*
T0*
shrink_axis_mask*
_output_shapes
: 
~
<input_layer/dense_input_layer/input_layer/I1/Reshape/shape/1Const*
dtype0*
value	B :*
_output_shapes
: 
к
:input_layer/dense_input_layer/input_layer/I1/Reshape/shapePack:input_layer/dense_input_layer/input_layer/I1/strided_slice<input_layer/dense_input_layer/input_layer/I1/Reshape/shape/1*
N*
_output_shapes
:*
T0
ж
4input_layer/dense_input_layer/input_layer/I1/ReshapeReshape7input_layer/dense_input_layer/input_layer/I1/ExpandDims:input_layer/dense_input_layer/input_layer/I1/Reshape/shape*'
_output_shapes
:€€€€€€€€€*
T0
З
<input_layer/dense_input_layer/input_layer/I10/ExpandDims/dimConst*
_output_shapes
: *
valueB :
€€€€€€€€€*
dtype0
≈
8input_layer/dense_input_layer/input_layer/I10/ExpandDims
ExpandDimsPlaceholder_9<input_layer/dense_input_layer/input_layer/I10/ExpandDims/dim*'
_output_shapes
:€€€€€€€€€*
T0
Ы
3input_layer/dense_input_layer/input_layer/I10/ShapeShape8input_layer/dense_input_layer/input_layer/I10/ExpandDims*
T0*
_output_shapes
:
Л
Ainput_layer/dense_input_layer/input_layer/I10/strided_slice/stackConst*
_output_shapes
:*
valueB: *
dtype0
Н
Cinput_layer/dense_input_layer/input_layer/I10/strided_slice/stack_1Const*
valueB:*
_output_shapes
:*
dtype0
Н
Cinput_layer/dense_input_layer/input_layer/I10/strided_slice/stack_2Const*
valueB:*
_output_shapes
:*
dtype0
У
;input_layer/dense_input_layer/input_layer/I10/strided_sliceStridedSlice3input_layer/dense_input_layer/input_layer/I10/ShapeAinput_layer/dense_input_layer/input_layer/I10/strided_slice/stackCinput_layer/dense_input_layer/input_layer/I10/strided_slice/stack_1Cinput_layer/dense_input_layer/input_layer/I10/strided_slice/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask

=input_layer/dense_input_layer/input_layer/I10/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
н
;input_layer/dense_input_layer/input_layer/I10/Reshape/shapePack;input_layer/dense_input_layer/input_layer/I10/strided_slice=input_layer/dense_input_layer/input_layer/I10/Reshape/shape/1*
T0*
_output_shapes
:*
N
й
5input_layer/dense_input_layer/input_layer/I10/ReshapeReshape8input_layer/dense_input_layer/input_layer/I10/ExpandDims;input_layer/dense_input_layer/input_layer/I10/Reshape/shape*'
_output_shapes
:€€€€€€€€€*
T0
З
<input_layer/dense_input_layer/input_layer/I11/ExpandDims/dimConst*
valueB :
€€€€€€€€€*
_output_shapes
: *
dtype0
∆
8input_layer/dense_input_layer/input_layer/I11/ExpandDims
ExpandDimsPlaceholder_10<input_layer/dense_input_layer/input_layer/I11/ExpandDims/dim*
T0*'
_output_shapes
:€€€€€€€€€
Ы
3input_layer/dense_input_layer/input_layer/I11/ShapeShape8input_layer/dense_input_layer/input_layer/I11/ExpandDims*
T0*
_output_shapes
:
Л
Ainput_layer/dense_input_layer/input_layer/I11/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Н
Cinput_layer/dense_input_layer/input_layer/I11/strided_slice/stack_1Const*
dtype0*
valueB:*
_output_shapes
:
Н
Cinput_layer/dense_input_layer/input_layer/I11/strided_slice/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
У
;input_layer/dense_input_layer/input_layer/I11/strided_sliceStridedSlice3input_layer/dense_input_layer/input_layer/I11/ShapeAinput_layer/dense_input_layer/input_layer/I11/strided_slice/stackCinput_layer/dense_input_layer/input_layer/I11/strided_slice/stack_1Cinput_layer/dense_input_layer/input_layer/I11/strided_slice/stack_2*
T0*
_output_shapes
: *
Index0*
shrink_axis_mask

=input_layer/dense_input_layer/input_layer/I11/Reshape/shape/1Const*
dtype0*
value	B :*
_output_shapes
: 
н
;input_layer/dense_input_layer/input_layer/I11/Reshape/shapePack;input_layer/dense_input_layer/input_layer/I11/strided_slice=input_layer/dense_input_layer/input_layer/I11/Reshape/shape/1*
T0*
N*
_output_shapes
:
й
5input_layer/dense_input_layer/input_layer/I11/ReshapeReshape8input_layer/dense_input_layer/input_layer/I11/ExpandDims;input_layer/dense_input_layer/input_layer/I11/Reshape/shape*
T0*'
_output_shapes
:€€€€€€€€€
З
<input_layer/dense_input_layer/input_layer/I12/ExpandDims/dimConst*
valueB :
€€€€€€€€€*
dtype0*
_output_shapes
: 
∆
8input_layer/dense_input_layer/input_layer/I12/ExpandDims
ExpandDimsPlaceholder_11<input_layer/dense_input_layer/input_layer/I12/ExpandDims/dim*'
_output_shapes
:€€€€€€€€€*
T0
Ы
3input_layer/dense_input_layer/input_layer/I12/ShapeShape8input_layer/dense_input_layer/input_layer/I12/ExpandDims*
T0*
_output_shapes
:
Л
Ainput_layer/dense_input_layer/input_layer/I12/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
Н
Cinput_layer/dense_input_layer/input_layer/I12/strided_slice/stack_1Const*
valueB:*
_output_shapes
:*
dtype0
Н
Cinput_layer/dense_input_layer/input_layer/I12/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
У
;input_layer/dense_input_layer/input_layer/I12/strided_sliceStridedSlice3input_layer/dense_input_layer/input_layer/I12/ShapeAinput_layer/dense_input_layer/input_layer/I12/strided_slice/stackCinput_layer/dense_input_layer/input_layer/I12/strided_slice/stack_1Cinput_layer/dense_input_layer/input_layer/I12/strided_slice/stack_2*
T0*
shrink_axis_mask*
Index0*
_output_shapes
: 

=input_layer/dense_input_layer/input_layer/I12/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
н
;input_layer/dense_input_layer/input_layer/I12/Reshape/shapePack;input_layer/dense_input_layer/input_layer/I12/strided_slice=input_layer/dense_input_layer/input_layer/I12/Reshape/shape/1*
_output_shapes
:*
T0*
N
й
5input_layer/dense_input_layer/input_layer/I12/ReshapeReshape8input_layer/dense_input_layer/input_layer/I12/ExpandDims;input_layer/dense_input_layer/input_layer/I12/Reshape/shape*
T0*'
_output_shapes
:€€€€€€€€€
З
<input_layer/dense_input_layer/input_layer/I13/ExpandDims/dimConst*
dtype0*
valueB :
€€€€€€€€€*
_output_shapes
: 
∆
8input_layer/dense_input_layer/input_layer/I13/ExpandDims
ExpandDimsPlaceholder_12<input_layer/dense_input_layer/input_layer/I13/ExpandDims/dim*'
_output_shapes
:€€€€€€€€€*
T0
Ы
3input_layer/dense_input_layer/input_layer/I13/ShapeShape8input_layer/dense_input_layer/input_layer/I13/ExpandDims*
_output_shapes
:*
T0
Л
Ainput_layer/dense_input_layer/input_layer/I13/strided_slice/stackConst*
_output_shapes
:*
valueB: *
dtype0
Н
Cinput_layer/dense_input_layer/input_layer/I13/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Н
Cinput_layer/dense_input_layer/input_layer/I13/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
У
;input_layer/dense_input_layer/input_layer/I13/strided_sliceStridedSlice3input_layer/dense_input_layer/input_layer/I13/ShapeAinput_layer/dense_input_layer/input_layer/I13/strided_slice/stackCinput_layer/dense_input_layer/input_layer/I13/strided_slice/stack_1Cinput_layer/dense_input_layer/input_layer/I13/strided_slice/stack_2*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: 

=input_layer/dense_input_layer/input_layer/I13/Reshape/shape/1Const*
value	B :*
dtype0*
_output_shapes
: 
н
;input_layer/dense_input_layer/input_layer/I13/Reshape/shapePack;input_layer/dense_input_layer/input_layer/I13/strided_slice=input_layer/dense_input_layer/input_layer/I13/Reshape/shape/1*
T0*
N*
_output_shapes
:
й
5input_layer/dense_input_layer/input_layer/I13/ReshapeReshape8input_layer/dense_input_layer/input_layer/I13/ExpandDims;input_layer/dense_input_layer/input_layer/I13/Reshape/shape*
T0*'
_output_shapes
:€€€€€€€€€
Ж
;input_layer/dense_input_layer/input_layer/I2/ExpandDims/dimConst*
dtype0*
_output_shapes
: *
valueB :
€€€€€€€€€
√
7input_layer/dense_input_layer/input_layer/I2/ExpandDims
ExpandDimsPlaceholder_1;input_layer/dense_input_layer/input_layer/I2/ExpandDims/dim*'
_output_shapes
:€€€€€€€€€*
T0
Щ
2input_layer/dense_input_layer/input_layer/I2/ShapeShape7input_layer/dense_input_layer/input_layer/I2/ExpandDims*
_output_shapes
:*
T0
К
@input_layer/dense_input_layer/input_layer/I2/strided_slice/stackConst*
dtype0*
valueB: *
_output_shapes
:
М
Binput_layer/dense_input_layer/input_layer/I2/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
М
Binput_layer/dense_input_layer/input_layer/I2/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
О
:input_layer/dense_input_layer/input_layer/I2/strided_sliceStridedSlice2input_layer/dense_input_layer/input_layer/I2/Shape@input_layer/dense_input_layer/input_layer/I2/strided_slice/stackBinput_layer/dense_input_layer/input_layer/I2/strided_slice/stack_1Binput_layer/dense_input_layer/input_layer/I2/strided_slice/stack_2*
_output_shapes
: *
shrink_axis_mask*
T0*
Index0
~
<input_layer/dense_input_layer/input_layer/I2/Reshape/shape/1Const*
value	B :*
dtype0*
_output_shapes
: 
к
:input_layer/dense_input_layer/input_layer/I2/Reshape/shapePack:input_layer/dense_input_layer/input_layer/I2/strided_slice<input_layer/dense_input_layer/input_layer/I2/Reshape/shape/1*
T0*
_output_shapes
:*
N
ж
4input_layer/dense_input_layer/input_layer/I2/ReshapeReshape7input_layer/dense_input_layer/input_layer/I2/ExpandDims:input_layer/dense_input_layer/input_layer/I2/Reshape/shape*
T0*'
_output_shapes
:€€€€€€€€€
Ж
;input_layer/dense_input_layer/input_layer/I3/ExpandDims/dimConst*
dtype0*
valueB :
€€€€€€€€€*
_output_shapes
: 
√
7input_layer/dense_input_layer/input_layer/I3/ExpandDims
ExpandDimsPlaceholder_2;input_layer/dense_input_layer/input_layer/I3/ExpandDims/dim*
T0*'
_output_shapes
:€€€€€€€€€
Щ
2input_layer/dense_input_layer/input_layer/I3/ShapeShape7input_layer/dense_input_layer/input_layer/I3/ExpandDims*
T0*
_output_shapes
:
К
@input_layer/dense_input_layer/input_layer/I3/strided_slice/stackConst*
dtype0*
valueB: *
_output_shapes
:
М
Binput_layer/dense_input_layer/input_layer/I3/strided_slice/stack_1Const*
valueB:*
_output_shapes
:*
dtype0
М
Binput_layer/dense_input_layer/input_layer/I3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
О
:input_layer/dense_input_layer/input_layer/I3/strided_sliceStridedSlice2input_layer/dense_input_layer/input_layer/I3/Shape@input_layer/dense_input_layer/input_layer/I3/strided_slice/stackBinput_layer/dense_input_layer/input_layer/I3/strided_slice/stack_1Binput_layer/dense_input_layer/input_layer/I3/strided_slice/stack_2*
_output_shapes
: *
Index0*
shrink_axis_mask*
T0
~
<input_layer/dense_input_layer/input_layer/I3/Reshape/shape/1Const*
_output_shapes
: *
value	B :*
dtype0
к
:input_layer/dense_input_layer/input_layer/I3/Reshape/shapePack:input_layer/dense_input_layer/input_layer/I3/strided_slice<input_layer/dense_input_layer/input_layer/I3/Reshape/shape/1*
T0*
_output_shapes
:*
N
ж
4input_layer/dense_input_layer/input_layer/I3/ReshapeReshape7input_layer/dense_input_layer/input_layer/I3/ExpandDims:input_layer/dense_input_layer/input_layer/I3/Reshape/shape*
T0*'
_output_shapes
:€€€€€€€€€
Ж
;input_layer/dense_input_layer/input_layer/I4/ExpandDims/dimConst*
dtype0*
valueB :
€€€€€€€€€*
_output_shapes
: 
√
7input_layer/dense_input_layer/input_layer/I4/ExpandDims
ExpandDimsPlaceholder_3;input_layer/dense_input_layer/input_layer/I4/ExpandDims/dim*
T0*'
_output_shapes
:€€€€€€€€€
Щ
2input_layer/dense_input_layer/input_layer/I4/ShapeShape7input_layer/dense_input_layer/input_layer/I4/ExpandDims*
_output_shapes
:*
T0
К
@input_layer/dense_input_layer/input_layer/I4/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB: 
М
Binput_layer/dense_input_layer/input_layer/I4/strided_slice/stack_1Const*
dtype0*
valueB:*
_output_shapes
:
М
Binput_layer/dense_input_layer/input_layer/I4/strided_slice/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
О
:input_layer/dense_input_layer/input_layer/I4/strided_sliceStridedSlice2input_layer/dense_input_layer/input_layer/I4/Shape@input_layer/dense_input_layer/input_layer/I4/strided_slice/stackBinput_layer/dense_input_layer/input_layer/I4/strided_slice/stack_1Binput_layer/dense_input_layer/input_layer/I4/strided_slice/stack_2*
T0*
shrink_axis_mask*
Index0*
_output_shapes
: 
~
<input_layer/dense_input_layer/input_layer/I4/Reshape/shape/1Const*
_output_shapes
: *
value	B :*
dtype0
к
:input_layer/dense_input_layer/input_layer/I4/Reshape/shapePack:input_layer/dense_input_layer/input_layer/I4/strided_slice<input_layer/dense_input_layer/input_layer/I4/Reshape/shape/1*
_output_shapes
:*
N*
T0
ж
4input_layer/dense_input_layer/input_layer/I4/ReshapeReshape7input_layer/dense_input_layer/input_layer/I4/ExpandDims:input_layer/dense_input_layer/input_layer/I4/Reshape/shape*
T0*'
_output_shapes
:€€€€€€€€€
Ж
;input_layer/dense_input_layer/input_layer/I5/ExpandDims/dimConst*
dtype0*
valueB :
€€€€€€€€€*
_output_shapes
: 
√
7input_layer/dense_input_layer/input_layer/I5/ExpandDims
ExpandDimsPlaceholder_4;input_layer/dense_input_layer/input_layer/I5/ExpandDims/dim*'
_output_shapes
:€€€€€€€€€*
T0
Щ
2input_layer/dense_input_layer/input_layer/I5/ShapeShape7input_layer/dense_input_layer/input_layer/I5/ExpandDims*
T0*
_output_shapes
:
К
@input_layer/dense_input_layer/input_layer/I5/strided_slice/stackConst*
_output_shapes
:*
valueB: *
dtype0
М
Binput_layer/dense_input_layer/input_layer/I5/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
М
Binput_layer/dense_input_layer/input_layer/I5/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
О
:input_layer/dense_input_layer/input_layer/I5/strided_sliceStridedSlice2input_layer/dense_input_layer/input_layer/I5/Shape@input_layer/dense_input_layer/input_layer/I5/strided_slice/stackBinput_layer/dense_input_layer/input_layer/I5/strided_slice/stack_1Binput_layer/dense_input_layer/input_layer/I5/strided_slice/stack_2*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: 
~
<input_layer/dense_input_layer/input_layer/I5/Reshape/shape/1Const*
value	B :*
_output_shapes
: *
dtype0
к
:input_layer/dense_input_layer/input_layer/I5/Reshape/shapePack:input_layer/dense_input_layer/input_layer/I5/strided_slice<input_layer/dense_input_layer/input_layer/I5/Reshape/shape/1*
_output_shapes
:*
T0*
N
ж
4input_layer/dense_input_layer/input_layer/I5/ReshapeReshape7input_layer/dense_input_layer/input_layer/I5/ExpandDims:input_layer/dense_input_layer/input_layer/I5/Reshape/shape*'
_output_shapes
:€€€€€€€€€*
T0
Ж
;input_layer/dense_input_layer/input_layer/I6/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€
√
7input_layer/dense_input_layer/input_layer/I6/ExpandDims
ExpandDimsPlaceholder_5;input_layer/dense_input_layer/input_layer/I6/ExpandDims/dim*'
_output_shapes
:€€€€€€€€€*
T0
Щ
2input_layer/dense_input_layer/input_layer/I6/ShapeShape7input_layer/dense_input_layer/input_layer/I6/ExpandDims*
_output_shapes
:*
T0
К
@input_layer/dense_input_layer/input_layer/I6/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
М
Binput_layer/dense_input_layer/input_layer/I6/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
М
Binput_layer/dense_input_layer/input_layer/I6/strided_slice/stack_2Const*
valueB:*
_output_shapes
:*
dtype0
О
:input_layer/dense_input_layer/input_layer/I6/strided_sliceStridedSlice2input_layer/dense_input_layer/input_layer/I6/Shape@input_layer/dense_input_layer/input_layer/I6/strided_slice/stackBinput_layer/dense_input_layer/input_layer/I6/strided_slice/stack_1Binput_layer/dense_input_layer/input_layer/I6/strided_slice/stack_2*
_output_shapes
: *
Index0*
shrink_axis_mask*
T0
~
<input_layer/dense_input_layer/input_layer/I6/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
к
:input_layer/dense_input_layer/input_layer/I6/Reshape/shapePack:input_layer/dense_input_layer/input_layer/I6/strided_slice<input_layer/dense_input_layer/input_layer/I6/Reshape/shape/1*
_output_shapes
:*
T0*
N
ж
4input_layer/dense_input_layer/input_layer/I6/ReshapeReshape7input_layer/dense_input_layer/input_layer/I6/ExpandDims:input_layer/dense_input_layer/input_layer/I6/Reshape/shape*'
_output_shapes
:€€€€€€€€€*
T0
Ж
;input_layer/dense_input_layer/input_layer/I7/ExpandDims/dimConst*
dtype0*
valueB :
€€€€€€€€€*
_output_shapes
: 
√
7input_layer/dense_input_layer/input_layer/I7/ExpandDims
ExpandDimsPlaceholder_6;input_layer/dense_input_layer/input_layer/I7/ExpandDims/dim*'
_output_shapes
:€€€€€€€€€*
T0
Щ
2input_layer/dense_input_layer/input_layer/I7/ShapeShape7input_layer/dense_input_layer/input_layer/I7/ExpandDims*
_output_shapes
:*
T0
К
@input_layer/dense_input_layer/input_layer/I7/strided_slice/stackConst*
dtype0*
valueB: *
_output_shapes
:
М
Binput_layer/dense_input_layer/input_layer/I7/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
М
Binput_layer/dense_input_layer/input_layer/I7/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
О
:input_layer/dense_input_layer/input_layer/I7/strided_sliceStridedSlice2input_layer/dense_input_layer/input_layer/I7/Shape@input_layer/dense_input_layer/input_layer/I7/strided_slice/stackBinput_layer/dense_input_layer/input_layer/I7/strided_slice/stack_1Binput_layer/dense_input_layer/input_layer/I7/strided_slice/stack_2*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: 
~
<input_layer/dense_input_layer/input_layer/I7/Reshape/shape/1Const*
_output_shapes
: *
value	B :*
dtype0
к
:input_layer/dense_input_layer/input_layer/I7/Reshape/shapePack:input_layer/dense_input_layer/input_layer/I7/strided_slice<input_layer/dense_input_layer/input_layer/I7/Reshape/shape/1*
N*
T0*
_output_shapes
:
ж
4input_layer/dense_input_layer/input_layer/I7/ReshapeReshape7input_layer/dense_input_layer/input_layer/I7/ExpandDims:input_layer/dense_input_layer/input_layer/I7/Reshape/shape*
T0*'
_output_shapes
:€€€€€€€€€
Ж
;input_layer/dense_input_layer/input_layer/I8/ExpandDims/dimConst*
dtype0*
valueB :
€€€€€€€€€*
_output_shapes
: 
√
7input_layer/dense_input_layer/input_layer/I8/ExpandDims
ExpandDimsPlaceholder_7;input_layer/dense_input_layer/input_layer/I8/ExpandDims/dim*
T0*'
_output_shapes
:€€€€€€€€€
Щ
2input_layer/dense_input_layer/input_layer/I8/ShapeShape7input_layer/dense_input_layer/input_layer/I8/ExpandDims*
T0*
_output_shapes
:
К
@input_layer/dense_input_layer/input_layer/I8/strided_slice/stackConst*
_output_shapes
:*
valueB: *
dtype0
М
Binput_layer/dense_input_layer/input_layer/I8/strided_slice/stack_1Const*
_output_shapes
:*
valueB:*
dtype0
М
Binput_layer/dense_input_layer/input_layer/I8/strided_slice/stack_2Const*
_output_shapes
:*
valueB:*
dtype0
О
:input_layer/dense_input_layer/input_layer/I8/strided_sliceStridedSlice2input_layer/dense_input_layer/input_layer/I8/Shape@input_layer/dense_input_layer/input_layer/I8/strided_slice/stackBinput_layer/dense_input_layer/input_layer/I8/strided_slice/stack_1Binput_layer/dense_input_layer/input_layer/I8/strided_slice/stack_2*
shrink_axis_mask*
_output_shapes
: *
T0*
Index0
~
<input_layer/dense_input_layer/input_layer/I8/Reshape/shape/1Const*
value	B :*
_output_shapes
: *
dtype0
к
:input_layer/dense_input_layer/input_layer/I8/Reshape/shapePack:input_layer/dense_input_layer/input_layer/I8/strided_slice<input_layer/dense_input_layer/input_layer/I8/Reshape/shape/1*
N*
_output_shapes
:*
T0
ж
4input_layer/dense_input_layer/input_layer/I8/ReshapeReshape7input_layer/dense_input_layer/input_layer/I8/ExpandDims:input_layer/dense_input_layer/input_layer/I8/Reshape/shape*'
_output_shapes
:€€€€€€€€€*
T0
Ж
;input_layer/dense_input_layer/input_layer/I9/ExpandDims/dimConst*
dtype0*
_output_shapes
: *
valueB :
€€€€€€€€€
√
7input_layer/dense_input_layer/input_layer/I9/ExpandDims
ExpandDimsPlaceholder_8;input_layer/dense_input_layer/input_layer/I9/ExpandDims/dim*
T0*'
_output_shapes
:€€€€€€€€€
Щ
2input_layer/dense_input_layer/input_layer/I9/ShapeShape7input_layer/dense_input_layer/input_layer/I9/ExpandDims*
_output_shapes
:*
T0
К
@input_layer/dense_input_layer/input_layer/I9/strided_slice/stackConst*
_output_shapes
:*
valueB: *
dtype0
М
Binput_layer/dense_input_layer/input_layer/I9/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
М
Binput_layer/dense_input_layer/input_layer/I9/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
О
:input_layer/dense_input_layer/input_layer/I9/strided_sliceStridedSlice2input_layer/dense_input_layer/input_layer/I9/Shape@input_layer/dense_input_layer/input_layer/I9/strided_slice/stackBinput_layer/dense_input_layer/input_layer/I9/strided_slice/stack_1Binput_layer/dense_input_layer/input_layer/I9/strided_slice/stack_2*
shrink_axis_mask*
_output_shapes
: *
Index0*
T0
~
<input_layer/dense_input_layer/input_layer/I9/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
к
:input_layer/dense_input_layer/input_layer/I9/Reshape/shapePack:input_layer/dense_input_layer/input_layer/I9/strided_slice<input_layer/dense_input_layer/input_layer/I9/Reshape/shape/1*
N*
T0*
_output_shapes
:
ж
4input_layer/dense_input_layer/input_layer/I9/ReshapeReshape7input_layer/dense_input_layer/input_layer/I9/ExpandDims:input_layer/dense_input_layer/input_layer/I9/Reshape/shape*'
_output_shapes
:€€€€€€€€€*
T0
w
5input_layer/dense_input_layer/input_layer/concat/axisConst*
_output_shapes
: *
value	B :*
dtype0
р
0input_layer/dense_input_layer/input_layer/concatConcatV24input_layer/dense_input_layer/input_layer/I1/Reshape5input_layer/dense_input_layer/input_layer/I10/Reshape5input_layer/dense_input_layer/input_layer/I11/Reshape5input_layer/dense_input_layer/input_layer/I12/Reshape5input_layer/dense_input_layer/input_layer/I13/Reshape4input_layer/dense_input_layer/input_layer/I2/Reshape4input_layer/dense_input_layer/input_layer/I3/Reshape4input_layer/dense_input_layer/input_layer/I4/Reshape4input_layer/dense_input_layer/input_layer/I5/Reshape4input_layer/dense_input_layer/input_layer/I6/Reshape4input_layer/dense_input_layer/input_layer/I7/Reshape4input_layer/dense_input_layer/input_layer/I8/Reshape4input_layer/dense_input_layer/input_layer/I9/Reshape5input_layer/dense_input_layer/input_layer/concat/axis*
N*'
_output_shapes
:€€€€€€€€€*
T0
Т
Ginput_layer/sparse_input_layer/input_layer/C10_embedding/ExpandDims/dimConst*
_output_shapes
: *
valueB :
€€€€€€€€€*
dtype0
№
Cinput_layer/sparse_input_layer/input_layer/C10_embedding/ExpandDims
ExpandDimsPlaceholder_22Ginput_layer/sparse_input_layer/input_layer/C10_embedding/ExpandDims/dim*'
_output_shapes
:€€€€€€€€€*
T0
Ш
Winput_layer/sparse_input_layer/input_layer/C10_embedding/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB B 
≠
Qinput_layer/sparse_input_layer/input_layer/C10_embedding/to_sparse_input/NotEqualNotEqualCinput_layer/sparse_input_layer/input_layer/C10_embedding/ExpandDimsWinput_layer/sparse_input_layer/input_layer/C10_embedding/to_sparse_input/ignore_value/x*
T0*'
_output_shapes
:€€€€€€€€€
’
Pinput_layer/sparse_input_layer/input_layer/C10_embedding/to_sparse_input/indicesWhereQinput_layer/sparse_input_layer/input_layer/C10_embedding/to_sparse_input/NotEqual*'
_output_shapes
:€€€€€€€€€
ґ
Oinput_layer/sparse_input_layer/input_layer/C10_embedding/to_sparse_input/valuesGatherNdCinput_layer/sparse_input_layer/input_layer/C10_embedding/ExpandDimsPinput_layer/sparse_input_layer/input_layer/C10_embedding/to_sparse_input/indices*
Tindices0	*
Tparams0*#
_output_shapes
:€€€€€€€€€
„
Tinput_layer/sparse_input_layer/input_layer/C10_embedding/to_sparse_input/dense_shapeShapeCinput_layer/sparse_input_layer/input_layer/C10_embedding/ExpandDims*
_output_shapes
:*
T0*
out_type0	
г
?input_layer/sparse_input_layer/input_layer/C10_embedding/lookupStringToHashBucketFastOinput_layer/sparse_input_layer/input_layer/C10_embedding/to_sparse_input/values*#
_output_shapes
:€€€€€€€€€*
num_bucketsРN
Э
minput_layer/sparse_input_layer/input_layer/C10_embedding/embedding_weights/Initializer/truncated_normal/shapeConst*
dtype0*
valueB"'     *
_output_shapes
:*]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C10_embedding/embedding_weights
Р
linput_layer/sparse_input_layer/input_layer/C10_embedding/embedding_weights/Initializer/truncated_normal/meanConst*
dtype0*]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C10_embedding/embedding_weights*
_output_shapes
: *
valueB
 *    
Т
ninput_layer/sparse_input_layer/input_layer/C10_embedding/embedding_weights/Initializer/truncated_normal/stddevConst*
_output_shapes
: *
dtype0*]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C10_embedding/embedding_weights*
valueB
 *  А>
П
winput_layer/sparse_input_layer/input_layer/C10_embedding/embedding_weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormalminput_layer/sparse_input_layer/input_layer/C10_embedding/embedding_weights/Initializer/truncated_normal/shape*
dtype0*
T0*
_output_shapes
:	РN*]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C10_embedding/embedding_weights
д
kinput_layer/sparse_input_layer/input_layer/C10_embedding/embedding_weights/Initializer/truncated_normal/mulMulwinput_layer/sparse_input_layer/input_layer/C10_embedding/embedding_weights/Initializer/truncated_normal/TruncatedNormalninput_layer/sparse_input_layer/input_layer/C10_embedding/embedding_weights/Initializer/truncated_normal/stddev*
T0*]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C10_embedding/embedding_weights*
_output_shapes
:	РN
“
ginput_layer/sparse_input_layer/input_layer/C10_embedding/embedding_weights/Initializer/truncated_normalAddkinput_layer/sparse_input_layer/input_layer/C10_embedding/embedding_weights/Initializer/truncated_normal/mullinput_layer/sparse_input_layer/input_layer/C10_embedding/embedding_weights/Initializer/truncated_normal/mean*]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C10_embedding/embedding_weights*
_output_shapes
:	РN*
T0
ы
Jinput_layer/sparse_input_layer/input_layer/C10_embedding/embedding_weights
VariableV2*
shape:	РN*
_output_shapes
:	РN*
dtype0*]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C10_embedding/embedding_weights
Щ
Qinput_layer/sparse_input_layer/input_layer/C10_embedding/embedding_weights/AssignAssignJinput_layer/sparse_input_layer/input_layer/C10_embedding/embedding_weightsginput_layer/sparse_input_layer/input_layer/C10_embedding/embedding_weights/Initializer/truncated_normal*]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C10_embedding/embedding_weights*
T0*
_output_shapes
:	РN
∞
Oinput_layer/sparse_input_layer/input_layer/C10_embedding/embedding_weights/readIdentityJinput_layer/sparse_input_layer/input_layer/C10_embedding/embedding_weights*
_output_shapes
:	РN*]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C10_embedding/embedding_weights*
T0
§
Zinput_layer/sparse_input_layer/input_layer/C10_embedding/C10_embedding_weights/Slice/beginConst*
_output_shapes
:*
valueB: *
dtype0
£
Yinput_layer/sparse_input_layer/input_layer/C10_embedding/C10_embedding_weights/Slice/sizeConst*
valueB:*
_output_shapes
:*
dtype0
Ь
Tinput_layer/sparse_input_layer/input_layer/C10_embedding/C10_embedding_weights/SliceSliceTinput_layer/sparse_input_layer/input_layer/C10_embedding/to_sparse_input/dense_shapeZinput_layer/sparse_input_layer/input_layer/C10_embedding/C10_embedding_weights/Slice/beginYinput_layer/sparse_input_layer/input_layer/C10_embedding/C10_embedding_weights/Slice/size*
T0	*
_output_shapes
:*
Index0
Ю
Tinput_layer/sparse_input_layer/input_layer/C10_embedding/C10_embedding_weights/ConstConst*
valueB: *
_output_shapes
:*
dtype0
®
Sinput_layer/sparse_input_layer/input_layer/C10_embedding/C10_embedding_weights/ProdProdTinput_layer/sparse_input_layer/input_layer/C10_embedding/C10_embedding_weights/SliceTinput_layer/sparse_input_layer/input_layer/C10_embedding/C10_embedding_weights/Const*
_output_shapes
: *
T0	
°
_input_layer/sparse_input_layer/input_layer/C10_embedding/C10_embedding_weights/GatherV2/indicesConst*
value	B :*
_output_shapes
: *
dtype0
Ю
\input_layer/sparse_input_layer/input_layer/C10_embedding/C10_embedding_weights/GatherV2/axisConst*
value	B : *
dtype0*
_output_shapes
: 
Љ
Winput_layer/sparse_input_layer/input_layer/C10_embedding/C10_embedding_weights/GatherV2GatherV2Tinput_layer/sparse_input_layer/input_layer/C10_embedding/to_sparse_input/dense_shape_input_layer/sparse_input_layer/input_layer/C10_embedding/C10_embedding_weights/GatherV2/indices\input_layer/sparse_input_layer/input_layer/C10_embedding/C10_embedding_weights/GatherV2/axis*
Tparams0	*
Taxis0*
_output_shapes
: *
Tindices0
є
Uinput_layer/sparse_input_layer/input_layer/C10_embedding/C10_embedding_weights/Cast/xPackSinput_layer/sparse_input_layer/input_layer/C10_embedding/C10_embedding_weights/ProdWinput_layer/sparse_input_layer/input_layer/C10_embedding/C10_embedding_weights/GatherV2*
N*
_output_shapes
:*
T0	
Ы
\input_layer/sparse_input_layer/input_layer/C10_embedding/C10_embedding_weights/SparseReshapeSparseReshapePinput_layer/sparse_input_layer/input_layer/C10_embedding/to_sparse_input/indicesTinput_layer/sparse_input_layer/input_layer/C10_embedding/to_sparse_input/dense_shapeUinput_layer/sparse_input_layer/input_layer/C10_embedding/C10_embedding_weights/Cast/x*-
_output_shapes
:€€€€€€€€€:
а
einput_layer/sparse_input_layer/input_layer/C10_embedding/C10_embedding_weights/SparseReshape/IdentityIdentity?input_layer/sparse_input_layer/input_layer/C10_embedding/lookup*
T0	*#
_output_shapes
:€€€€€€€€€
Я
]input_layer/sparse_input_layer/input_layer/C10_embedding/C10_embedding_weights/GreaterEqual/yConst*
_output_shapes
: *
value	B	 R *
dtype0	
я
[input_layer/sparse_input_layer/input_layer/C10_embedding/C10_embedding_weights/GreaterEqualGreaterEqualeinput_layer/sparse_input_layer/input_layer/C10_embedding/C10_embedding_weights/SparseReshape/Identity]input_layer/sparse_input_layer/input_layer/C10_embedding/C10_embedding_weights/GreaterEqual/y*
T0	*#
_output_shapes
:€€€€€€€€€
г
Tinput_layer/sparse_input_layer/input_layer/C10_embedding/C10_embedding_weights/WhereWhere[input_layer/sparse_input_layer/input_layer/C10_embedding/C10_embedding_weights/GreaterEqual*'
_output_shapes
:€€€€€€€€€
ѓ
\input_layer/sparse_input_layer/input_layer/C10_embedding/C10_embedding_weights/Reshape/shapeConst*
valueB:
€€€€€€€€€*
_output_shapes
:*
dtype0
√
Vinput_layer/sparse_input_layer/input_layer/C10_embedding/C10_embedding_weights/ReshapeReshapeTinput_layer/sparse_input_layer/input_layer/C10_embedding/C10_embedding_weights/Where\input_layer/sparse_input_layer/input_layer/C10_embedding/C10_embedding_weights/Reshape/shape*#
_output_shapes
:€€€€€€€€€*
T0	
†
^input_layer/sparse_input_layer/input_layer/C10_embedding/C10_embedding_weights/GatherV2_1/axisConst*
dtype0*
_output_shapes
: *
value	B : 
–
Yinput_layer/sparse_input_layer/input_layer/C10_embedding/C10_embedding_weights/GatherV2_1GatherV2\input_layer/sparse_input_layer/input_layer/C10_embedding/C10_embedding_weights/SparseReshapeVinput_layer/sparse_input_layer/input_layer/C10_embedding/C10_embedding_weights/Reshape^input_layer/sparse_input_layer/input_layer/C10_embedding/C10_embedding_weights/GatherV2_1/axis*'
_output_shapes
:€€€€€€€€€*
Tparams0	*
Tindices0	*
Taxis0
†
^input_layer/sparse_input_layer/input_layer/C10_embedding/C10_embedding_weights/GatherV2_2/axisConst*
_output_shapes
: *
value	B : *
dtype0
’
Yinput_layer/sparse_input_layer/input_layer/C10_embedding/C10_embedding_weights/GatherV2_2GatherV2einput_layer/sparse_input_layer/input_layer/C10_embedding/C10_embedding_weights/SparseReshape/IdentityVinput_layer/sparse_input_layer/input_layer/C10_embedding/C10_embedding_weights/Reshape^input_layer/sparse_input_layer/input_layer/C10_embedding/C10_embedding_weights/GatherV2_2/axis*
Taxis0*
Tindices0	*#
_output_shapes
:€€€€€€€€€*
Tparams0	
и
Winput_layer/sparse_input_layer/input_layer/C10_embedding/C10_embedding_weights/IdentityIdentity^input_layer/sparse_input_layer/input_layer/C10_embedding/C10_embedding_weights/SparseReshape:1*
_output_shapes
:*
T0	
™
hinput_layer/sparse_input_layer/input_layer/C10_embedding/C10_embedding_weights/SparseFillEmptyRows/ConstConst*
dtype0	*
_output_shapes
: *
value	B	 R 
е
vinput_layer/sparse_input_layer/input_layer/C10_embedding/C10_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowsSparseFillEmptyRowsYinput_layer/sparse_input_layer/input_layer/C10_embedding/C10_embedding_weights/GatherV2_1Yinput_layer/sparse_input_layer/input_layer/C10_embedding/C10_embedding_weights/GatherV2_2Winput_layer/sparse_input_layer/input_layer/C10_embedding/C10_embedding_weights/Identityhinput_layer/sparse_input_layer/input_layer/C10_embedding/C10_embedding_weights/SparseFillEmptyRows/Const*
T0	*T
_output_shapesB
@:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€
Ћ
zinput_layer/sparse_input_layer/input_layer/C10_embedding/C10_embedding_weights/embedding_lookup_sparse/strided_slice/stackConst*
dtype0*
valueB"        *
_output_shapes
:
Ќ
|input_layer/sparse_input_layer/input_layer/C10_embedding/C10_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1Const*
valueB"       *
dtype0*
_output_shapes
:
Ќ
|input_layer/sparse_input_layer/input_layer/C10_embedding/C10_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2Const*
_output_shapes
:*
valueB"      *
dtype0
й
tinput_layer/sparse_input_layer/input_layer/C10_embedding/C10_embedding_weights/embedding_lookup_sparse/strided_sliceStridedSlicevinput_layer/sparse_input_layer/input_layer/C10_embedding/C10_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowszinput_layer/sparse_input_layer/input_layer/C10_embedding/C10_embedding_weights/embedding_lookup_sparse/strided_slice/stack|input_layer/sparse_input_layer/input_layer/C10_embedding/C10_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1|input_layer/sparse_input_layer/input_layer/C10_embedding/C10_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2*
Index0*

begin_mask*
shrink_axis_mask*
T0	*
end_mask*#
_output_shapes
:€€€€€€€€€
¶
kinput_layer/sparse_input_layer/input_layer/C10_embedding/C10_embedding_weights/embedding_lookup_sparse/CastCasttinput_layer/sparse_input_layer/input_layer/C10_embedding/C10_embedding_weights/embedding_lookup_sparse/strided_slice*

DstT0*#
_output_shapes
:€€€€€€€€€*

SrcT0	
Ѓ
minput_layer/sparse_input_layer/input_layer/C10_embedding/C10_embedding_weights/embedding_lookup_sparse/UniqueUniquexinput_layer/sparse_input_layer/input_layer/C10_embedding/C10_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:1*
T0	*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
Э
|input_layer/sparse_input_layer/input_layer/C10_embedding/C10_embedding_weights/embedding_lookup_sparse/embedding_lookup/axisConst*
dtype0*
value	B : *]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C10_embedding/embedding_weights*
_output_shapes
: 
х
winput_layer/sparse_input_layer/input_layer/C10_embedding/C10_embedding_weights/embedding_lookup_sparse/embedding_lookupGatherV2Oinput_layer/sparse_input_layer/input_layer/C10_embedding/embedding_weights/readminput_layer/sparse_input_layer/input_layer/C10_embedding/C10_embedding_weights/embedding_lookup_sparse/Unique|input_layer/sparse_input_layer/input_layer/C10_embedding/C10_embedding_weights/embedding_lookup_sparse/embedding_lookup/axis*]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C10_embedding/embedding_weights*'
_output_shapes
:€€€€€€€€€*
Tindices0	*
Taxis0*
Tparams0
Є
Аinput_layer/sparse_input_layer/input_layer/C10_embedding/C10_embedding_weights/embedding_lookup_sparse/embedding_lookup/IdentityIdentitywinput_layer/sparse_input_layer/input_layer/C10_embedding/C10_embedding_weights/embedding_lookup_sparse/embedding_lookup*'
_output_shapes
:€€€€€€€€€*
T0
О
finput_layer/sparse_input_layer/input_layer/C10_embedding/C10_embedding_weights/embedding_lookup_sparseSparseSegmentMeanАinput_layer/sparse_input_layer/input_layer/C10_embedding/C10_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identityoinput_layer/sparse_input_layer/input_layer/C10_embedding/C10_embedding_weights/embedding_lookup_sparse/Unique:1kinput_layer/sparse_input_layer/input_layer/C10_embedding/C10_embedding_weights/embedding_lookup_sparse/Cast*
T0*'
_output_shapes
:€€€€€€€€€
ѓ
^input_layer/sparse_input_layer/input_layer/C10_embedding/C10_embedding_weights/Reshape_1/shapeConst*
valueB"€€€€   *
_output_shapes
:*
dtype0
п
Xinput_layer/sparse_input_layer/input_layer/C10_embedding/C10_embedding_weights/Reshape_1Reshapexinput_layer/sparse_input_layer/input_layer/C10_embedding/C10_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:2^input_layer/sparse_input_layer/input_layer/C10_embedding/C10_embedding_weights/Reshape_1/shape*'
_output_shapes
:€€€€€€€€€*
T0

к
Tinput_layer/sparse_input_layer/input_layer/C10_embedding/C10_embedding_weights/ShapeShapefinput_layer/sparse_input_layer/input_layer/C10_embedding/C10_embedding_weights/embedding_lookup_sparse*
_output_shapes
:*
T0
ђ
binput_layer/sparse_input_layer/input_layer/C10_embedding/C10_embedding_weights/strided_slice/stackConst*
_output_shapes
:*
valueB:*
dtype0
Ѓ
dinput_layer/sparse_input_layer/input_layer/C10_embedding/C10_embedding_weights/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
Ѓ
dinput_layer/sparse_input_layer/input_layer/C10_embedding/C10_embedding_weights/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
Є
\input_layer/sparse_input_layer/input_layer/C10_embedding/C10_embedding_weights/strided_sliceStridedSliceTinput_layer/sparse_input_layer/input_layer/C10_embedding/C10_embedding_weights/Shapebinput_layer/sparse_input_layer/input_layer/C10_embedding/C10_embedding_weights/strided_slice/stackdinput_layer/sparse_input_layer/input_layer/C10_embedding/C10_embedding_weights/strided_slice/stack_1dinput_layer/sparse_input_layer/input_layer/C10_embedding/C10_embedding_weights/strided_slice/stack_2*
T0*
Index0*
_output_shapes
: *
shrink_axis_mask
Ш
Vinput_layer/sparse_input_layer/input_layer/C10_embedding/C10_embedding_weights/stack/0Const*
value	B :*
_output_shapes
: *
dtype0
ј
Tinput_layer/sparse_input_layer/input_layer/C10_embedding/C10_embedding_weights/stackPackVinput_layer/sparse_input_layer/input_layer/C10_embedding/C10_embedding_weights/stack/0\input_layer/sparse_input_layer/input_layer/C10_embedding/C10_embedding_weights/strided_slice*
T0*
_output_shapes
:*
N
∆
Sinput_layer/sparse_input_layer/input_layer/C10_embedding/C10_embedding_weights/TileTileXinput_layer/sparse_input_layer/input_layer/C10_embedding/C10_embedding_weights/Reshape_1Tinput_layer/sparse_input_layer/input_layer/C10_embedding/C10_embedding_weights/stack*0
_output_shapes
:€€€€€€€€€€€€€€€€€€*
T0

А
Yinput_layer/sparse_input_layer/input_layer/C10_embedding/C10_embedding_weights/zeros_like	ZerosLikefinput_layer/sparse_input_layer/input_layer/C10_embedding/C10_embedding_weights/embedding_lookup_sparse*
T0*'
_output_shapes
:€€€€€€€€€
Ґ
Ninput_layer/sparse_input_layer/input_layer/C10_embedding/C10_embedding_weightsSelectSinput_layer/sparse_input_layer/input_layer/C10_embedding/C10_embedding_weights/TileYinput_layer/sparse_input_layer/input_layer/C10_embedding/C10_embedding_weights/zeros_likefinput_layer/sparse_input_layer/input_layer/C10_embedding/C10_embedding_weights/embedding_lookup_sparse*
T0*'
_output_shapes
:€€€€€€€€€
з
Uinput_layer/sparse_input_layer/input_layer/C10_embedding/C10_embedding_weights/Cast_1CastTinput_layer/sparse_input_layer/input_layer/C10_embedding/to_sparse_input/dense_shape*

DstT0*
_output_shapes
:*

SrcT0	
¶
\input_layer/sparse_input_layer/input_layer/C10_embedding/C10_embedding_weights/Slice_1/beginConst*
valueB: *
dtype0*
_output_shapes
:
•
[input_layer/sparse_input_layer/input_layer/C10_embedding/C10_embedding_weights/Slice_1/sizeConst*
valueB:*
dtype0*
_output_shapes
:
£
Vinput_layer/sparse_input_layer/input_layer/C10_embedding/C10_embedding_weights/Slice_1SliceUinput_layer/sparse_input_layer/input_layer/C10_embedding/C10_embedding_weights/Cast_1\input_layer/sparse_input_layer/input_layer/C10_embedding/C10_embedding_weights/Slice_1/begin[input_layer/sparse_input_layer/input_layer/C10_embedding/C10_embedding_weights/Slice_1/size*
Index0*
T0*
_output_shapes
:
‘
Vinput_layer/sparse_input_layer/input_layer/C10_embedding/C10_embedding_weights/Shape_1ShapeNinput_layer/sparse_input_layer/input_layer/C10_embedding/C10_embedding_weights*
_output_shapes
:*
T0
¶
\input_layer/sparse_input_layer/input_layer/C10_embedding/C10_embedding_weights/Slice_2/beginConst*
_output_shapes
:*
dtype0*
valueB:
Ѓ
[input_layer/sparse_input_layer/input_layer/C10_embedding/C10_embedding_weights/Slice_2/sizeConst*
_output_shapes
:*
valueB:
€€€€€€€€€*
dtype0
§
Vinput_layer/sparse_input_layer/input_layer/C10_embedding/C10_embedding_weights/Slice_2SliceVinput_layer/sparse_input_layer/input_layer/C10_embedding/C10_embedding_weights/Shape_1\input_layer/sparse_input_layer/input_layer/C10_embedding/C10_embedding_weights/Slice_2/begin[input_layer/sparse_input_layer/input_layer/C10_embedding/C10_embedding_weights/Slice_2/size*
T0*
Index0*
_output_shapes
:
Ь
Zinput_layer/sparse_input_layer/input_layer/C10_embedding/C10_embedding_weights/concat/axisConst*
dtype0*
_output_shapes
: *
value	B : 
Ы
Uinput_layer/sparse_input_layer/input_layer/C10_embedding/C10_embedding_weights/concatConcatV2Vinput_layer/sparse_input_layer/input_layer/C10_embedding/C10_embedding_weights/Slice_1Vinput_layer/sparse_input_layer/input_layer/C10_embedding/C10_embedding_weights/Slice_2Zinput_layer/sparse_input_layer/input_layer/C10_embedding/C10_embedding_weights/concat/axis*
T0*
_output_shapes
:*
N
Љ
Xinput_layer/sparse_input_layer/input_layer/C10_embedding/C10_embedding_weights/Reshape_2ReshapeNinput_layer/sparse_input_layer/input_layer/C10_embedding/C10_embedding_weightsUinput_layer/sparse_input_layer/input_layer/C10_embedding/C10_embedding_weights/concat*'
_output_shapes
:€€€€€€€€€*
T0
∆
>input_layer/sparse_input_layer/input_layer/C10_embedding/ShapeShapeXinput_layer/sparse_input_layer/input_layer/C10_embedding/C10_embedding_weights/Reshape_2*
_output_shapes
:*
T0
Ц
Linput_layer/sparse_input_layer/input_layer/C10_embedding/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
Ш
Ninput_layer/sparse_input_layer/input_layer/C10_embedding/strided_slice/stack_1Const*
dtype0*
valueB:*
_output_shapes
:
Ш
Ninput_layer/sparse_input_layer/input_layer/C10_embedding/strided_slice/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
 
Finput_layer/sparse_input_layer/input_layer/C10_embedding/strided_sliceStridedSlice>input_layer/sparse_input_layer/input_layer/C10_embedding/ShapeLinput_layer/sparse_input_layer/input_layer/C10_embedding/strided_slice/stackNinput_layer/sparse_input_layer/input_layer/C10_embedding/strided_slice/stack_1Ninput_layer/sparse_input_layer/input_layer/C10_embedding/strided_slice/stack_2*
Index0*
_output_shapes
: *
T0*
shrink_axis_mask
К
Hinput_layer/sparse_input_layer/input_layer/C10_embedding/Reshape/shape/1Const*
value	B :*
_output_shapes
: *
dtype0
О
Finput_layer/sparse_input_layer/input_layer/C10_embedding/Reshape/shapePackFinput_layer/sparse_input_layer/input_layer/C10_embedding/strided_sliceHinput_layer/sparse_input_layer/input_layer/C10_embedding/Reshape/shape/1*
N*
T0*
_output_shapes
:
Я
@input_layer/sparse_input_layer/input_layer/C10_embedding/ReshapeReshapeXinput_layer/sparse_input_layer/input_layer/C10_embedding/C10_embedding_weights/Reshape_2Finput_layer/sparse_input_layer/input_layer/C10_embedding/Reshape/shape*'
_output_shapes
:€€€€€€€€€*
T0
Т
Ginput_layer/sparse_input_layer/input_layer/C11_embedding/ExpandDims/dimConst*
dtype0*
valueB :
€€€€€€€€€*
_output_shapes
: 
№
Cinput_layer/sparse_input_layer/input_layer/C11_embedding/ExpandDims
ExpandDimsPlaceholder_23Ginput_layer/sparse_input_layer/input_layer/C11_embedding/ExpandDims/dim*'
_output_shapes
:€€€€€€€€€*
T0
Ш
Winput_layer/sparse_input_layer/input_layer/C11_embedding/to_sparse_input/ignore_value/xConst*
valueB B *
dtype0*
_output_shapes
: 
≠
Qinput_layer/sparse_input_layer/input_layer/C11_embedding/to_sparse_input/NotEqualNotEqualCinput_layer/sparse_input_layer/input_layer/C11_embedding/ExpandDimsWinput_layer/sparse_input_layer/input_layer/C11_embedding/to_sparse_input/ignore_value/x*
T0*'
_output_shapes
:€€€€€€€€€
’
Pinput_layer/sparse_input_layer/input_layer/C11_embedding/to_sparse_input/indicesWhereQinput_layer/sparse_input_layer/input_layer/C11_embedding/to_sparse_input/NotEqual*'
_output_shapes
:€€€€€€€€€
ґ
Oinput_layer/sparse_input_layer/input_layer/C11_embedding/to_sparse_input/valuesGatherNdCinput_layer/sparse_input_layer/input_layer/C11_embedding/ExpandDimsPinput_layer/sparse_input_layer/input_layer/C11_embedding/to_sparse_input/indices*
Tindices0	*
Tparams0*#
_output_shapes
:€€€€€€€€€
„
Tinput_layer/sparse_input_layer/input_layer/C11_embedding/to_sparse_input/dense_shapeShapeCinput_layer/sparse_input_layer/input_layer/C11_embedding/ExpandDims*
T0*
out_type0	*
_output_shapes
:
г
?input_layer/sparse_input_layer/input_layer/C11_embedding/lookupStringToHashBucketFastOinput_layer/sparse_input_layer/input_layer/C11_embedding/to_sparse_input/values*#
_output_shapes
:€€€€€€€€€*
num_bucketsРN
Э
minput_layer/sparse_input_layer/input_layer/C11_embedding/embedding_weights/Initializer/truncated_normal/shapeConst*
valueB"'     *
_output_shapes
:*]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C11_embedding/embedding_weights*
dtype0
Р
linput_layer/sparse_input_layer/input_layer/C11_embedding/embedding_weights/Initializer/truncated_normal/meanConst*]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C11_embedding/embedding_weights*
_output_shapes
: *
valueB
 *    *
dtype0
Т
ninput_layer/sparse_input_layer/input_layer/C11_embedding/embedding_weights/Initializer/truncated_normal/stddevConst*
dtype0*
_output_shapes
: *]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C11_embedding/embedding_weights*
valueB
 *  А>
П
winput_layer/sparse_input_layer/input_layer/C11_embedding/embedding_weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormalminput_layer/sparse_input_layer/input_layer/C11_embedding/embedding_weights/Initializer/truncated_normal/shape*
T0*
_output_shapes
:	РN*]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C11_embedding/embedding_weights*
dtype0
д
kinput_layer/sparse_input_layer/input_layer/C11_embedding/embedding_weights/Initializer/truncated_normal/mulMulwinput_layer/sparse_input_layer/input_layer/C11_embedding/embedding_weights/Initializer/truncated_normal/TruncatedNormalninput_layer/sparse_input_layer/input_layer/C11_embedding/embedding_weights/Initializer/truncated_normal/stddev*
_output_shapes
:	РN*]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C11_embedding/embedding_weights*
T0
“
ginput_layer/sparse_input_layer/input_layer/C11_embedding/embedding_weights/Initializer/truncated_normalAddkinput_layer/sparse_input_layer/input_layer/C11_embedding/embedding_weights/Initializer/truncated_normal/mullinput_layer/sparse_input_layer/input_layer/C11_embedding/embedding_weights/Initializer/truncated_normal/mean*
_output_shapes
:	РN*]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C11_embedding/embedding_weights*
T0
ы
Jinput_layer/sparse_input_layer/input_layer/C11_embedding/embedding_weights
VariableV2*
dtype0*
_output_shapes
:	РN*]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C11_embedding/embedding_weights*
shape:	РN
Щ
Qinput_layer/sparse_input_layer/input_layer/C11_embedding/embedding_weights/AssignAssignJinput_layer/sparse_input_layer/input_layer/C11_embedding/embedding_weightsginput_layer/sparse_input_layer/input_layer/C11_embedding/embedding_weights/Initializer/truncated_normal*]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C11_embedding/embedding_weights*
_output_shapes
:	РN*
T0
∞
Oinput_layer/sparse_input_layer/input_layer/C11_embedding/embedding_weights/readIdentityJinput_layer/sparse_input_layer/input_layer/C11_embedding/embedding_weights*
T0*
_output_shapes
:	РN*]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C11_embedding/embedding_weights
§
Zinput_layer/sparse_input_layer/input_layer/C11_embedding/C11_embedding_weights/Slice/beginConst*
_output_shapes
:*
dtype0*
valueB: 
£
Yinput_layer/sparse_input_layer/input_layer/C11_embedding/C11_embedding_weights/Slice/sizeConst*
dtype0*
valueB:*
_output_shapes
:
Ь
Tinput_layer/sparse_input_layer/input_layer/C11_embedding/C11_embedding_weights/SliceSliceTinput_layer/sparse_input_layer/input_layer/C11_embedding/to_sparse_input/dense_shapeZinput_layer/sparse_input_layer/input_layer/C11_embedding/C11_embedding_weights/Slice/beginYinput_layer/sparse_input_layer/input_layer/C11_embedding/C11_embedding_weights/Slice/size*
_output_shapes
:*
Index0*
T0	
Ю
Tinput_layer/sparse_input_layer/input_layer/C11_embedding/C11_embedding_weights/ConstConst*
dtype0*
_output_shapes
:*
valueB: 
®
Sinput_layer/sparse_input_layer/input_layer/C11_embedding/C11_embedding_weights/ProdProdTinput_layer/sparse_input_layer/input_layer/C11_embedding/C11_embedding_weights/SliceTinput_layer/sparse_input_layer/input_layer/C11_embedding/C11_embedding_weights/Const*
_output_shapes
: *
T0	
°
_input_layer/sparse_input_layer/input_layer/C11_embedding/C11_embedding_weights/GatherV2/indicesConst*
dtype0*
_output_shapes
: *
value	B :
Ю
\input_layer/sparse_input_layer/input_layer/C11_embedding/C11_embedding_weights/GatherV2/axisConst*
value	B : *
dtype0*
_output_shapes
: 
Љ
Winput_layer/sparse_input_layer/input_layer/C11_embedding/C11_embedding_weights/GatherV2GatherV2Tinput_layer/sparse_input_layer/input_layer/C11_embedding/to_sparse_input/dense_shape_input_layer/sparse_input_layer/input_layer/C11_embedding/C11_embedding_weights/GatherV2/indices\input_layer/sparse_input_layer/input_layer/C11_embedding/C11_embedding_weights/GatherV2/axis*
Taxis0*
_output_shapes
: *
Tparams0	*
Tindices0
є
Uinput_layer/sparse_input_layer/input_layer/C11_embedding/C11_embedding_weights/Cast/xPackSinput_layer/sparse_input_layer/input_layer/C11_embedding/C11_embedding_weights/ProdWinput_layer/sparse_input_layer/input_layer/C11_embedding/C11_embedding_weights/GatherV2*
T0	*
N*
_output_shapes
:
Ы
\input_layer/sparse_input_layer/input_layer/C11_embedding/C11_embedding_weights/SparseReshapeSparseReshapePinput_layer/sparse_input_layer/input_layer/C11_embedding/to_sparse_input/indicesTinput_layer/sparse_input_layer/input_layer/C11_embedding/to_sparse_input/dense_shapeUinput_layer/sparse_input_layer/input_layer/C11_embedding/C11_embedding_weights/Cast/x*-
_output_shapes
:€€€€€€€€€:
а
einput_layer/sparse_input_layer/input_layer/C11_embedding/C11_embedding_weights/SparseReshape/IdentityIdentity?input_layer/sparse_input_layer/input_layer/C11_embedding/lookup*#
_output_shapes
:€€€€€€€€€*
T0	
Я
]input_layer/sparse_input_layer/input_layer/C11_embedding/C11_embedding_weights/GreaterEqual/yConst*
_output_shapes
: *
value	B	 R *
dtype0	
я
[input_layer/sparse_input_layer/input_layer/C11_embedding/C11_embedding_weights/GreaterEqualGreaterEqualeinput_layer/sparse_input_layer/input_layer/C11_embedding/C11_embedding_weights/SparseReshape/Identity]input_layer/sparse_input_layer/input_layer/C11_embedding/C11_embedding_weights/GreaterEqual/y*
T0	*#
_output_shapes
:€€€€€€€€€
г
Tinput_layer/sparse_input_layer/input_layer/C11_embedding/C11_embedding_weights/WhereWhere[input_layer/sparse_input_layer/input_layer/C11_embedding/C11_embedding_weights/GreaterEqual*'
_output_shapes
:€€€€€€€€€
ѓ
\input_layer/sparse_input_layer/input_layer/C11_embedding/C11_embedding_weights/Reshape/shapeConst*
valueB:
€€€€€€€€€*
dtype0*
_output_shapes
:
√
Vinput_layer/sparse_input_layer/input_layer/C11_embedding/C11_embedding_weights/ReshapeReshapeTinput_layer/sparse_input_layer/input_layer/C11_embedding/C11_embedding_weights/Where\input_layer/sparse_input_layer/input_layer/C11_embedding/C11_embedding_weights/Reshape/shape*#
_output_shapes
:€€€€€€€€€*
T0	
†
^input_layer/sparse_input_layer/input_layer/C11_embedding/C11_embedding_weights/GatherV2_1/axisConst*
value	B : *
_output_shapes
: *
dtype0
–
Yinput_layer/sparse_input_layer/input_layer/C11_embedding/C11_embedding_weights/GatherV2_1GatherV2\input_layer/sparse_input_layer/input_layer/C11_embedding/C11_embedding_weights/SparseReshapeVinput_layer/sparse_input_layer/input_layer/C11_embedding/C11_embedding_weights/Reshape^input_layer/sparse_input_layer/input_layer/C11_embedding/C11_embedding_weights/GatherV2_1/axis*
Taxis0*
Tindices0	*'
_output_shapes
:€€€€€€€€€*
Tparams0	
†
^input_layer/sparse_input_layer/input_layer/C11_embedding/C11_embedding_weights/GatherV2_2/axisConst*
value	B : *
_output_shapes
: *
dtype0
’
Yinput_layer/sparse_input_layer/input_layer/C11_embedding/C11_embedding_weights/GatherV2_2GatherV2einput_layer/sparse_input_layer/input_layer/C11_embedding/C11_embedding_weights/SparseReshape/IdentityVinput_layer/sparse_input_layer/input_layer/C11_embedding/C11_embedding_weights/Reshape^input_layer/sparse_input_layer/input_layer/C11_embedding/C11_embedding_weights/GatherV2_2/axis*#
_output_shapes
:€€€€€€€€€*
Tparams0	*
Tindices0	*
Taxis0
и
Winput_layer/sparse_input_layer/input_layer/C11_embedding/C11_embedding_weights/IdentityIdentity^input_layer/sparse_input_layer/input_layer/C11_embedding/C11_embedding_weights/SparseReshape:1*
_output_shapes
:*
T0	
™
hinput_layer/sparse_input_layer/input_layer/C11_embedding/C11_embedding_weights/SparseFillEmptyRows/ConstConst*
dtype0	*
value	B	 R *
_output_shapes
: 
е
vinput_layer/sparse_input_layer/input_layer/C11_embedding/C11_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowsSparseFillEmptyRowsYinput_layer/sparse_input_layer/input_layer/C11_embedding/C11_embedding_weights/GatherV2_1Yinput_layer/sparse_input_layer/input_layer/C11_embedding/C11_embedding_weights/GatherV2_2Winput_layer/sparse_input_layer/input_layer/C11_embedding/C11_embedding_weights/Identityhinput_layer/sparse_input_layer/input_layer/C11_embedding/C11_embedding_weights/SparseFillEmptyRows/Const*T
_output_shapesB
@:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€*
T0	
Ћ
zinput_layer/sparse_input_layer/input_layer/C11_embedding/C11_embedding_weights/embedding_lookup_sparse/strided_slice/stackConst*
_output_shapes
:*
valueB"        *
dtype0
Ќ
|input_layer/sparse_input_layer/input_layer/C11_embedding/C11_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1Const*
dtype0*
valueB"       *
_output_shapes
:
Ќ
|input_layer/sparse_input_layer/input_layer/C11_embedding/C11_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
й
tinput_layer/sparse_input_layer/input_layer/C11_embedding/C11_embedding_weights/embedding_lookup_sparse/strided_sliceStridedSlicevinput_layer/sparse_input_layer/input_layer/C11_embedding/C11_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowszinput_layer/sparse_input_layer/input_layer/C11_embedding/C11_embedding_weights/embedding_lookup_sparse/strided_slice/stack|input_layer/sparse_input_layer/input_layer/C11_embedding/C11_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1|input_layer/sparse_input_layer/input_layer/C11_embedding/C11_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2*
Index0*
shrink_axis_mask*
T0	*
end_mask*#
_output_shapes
:€€€€€€€€€*

begin_mask
¶
kinput_layer/sparse_input_layer/input_layer/C11_embedding/C11_embedding_weights/embedding_lookup_sparse/CastCasttinput_layer/sparse_input_layer/input_layer/C11_embedding/C11_embedding_weights/embedding_lookup_sparse/strided_slice*#
_output_shapes
:€€€€€€€€€*

SrcT0	*

DstT0
Ѓ
minput_layer/sparse_input_layer/input_layer/C11_embedding/C11_embedding_weights/embedding_lookup_sparse/UniqueUniquexinput_layer/sparse_input_layer/input_layer/C11_embedding/C11_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€*
T0	
Э
|input_layer/sparse_input_layer/input_layer/C11_embedding/C11_embedding_weights/embedding_lookup_sparse/embedding_lookup/axisConst*
dtype0*
_output_shapes
: *]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C11_embedding/embedding_weights*
value	B : 
х
winput_layer/sparse_input_layer/input_layer/C11_embedding/C11_embedding_weights/embedding_lookup_sparse/embedding_lookupGatherV2Oinput_layer/sparse_input_layer/input_layer/C11_embedding/embedding_weights/readminput_layer/sparse_input_layer/input_layer/C11_embedding/C11_embedding_weights/embedding_lookup_sparse/Unique|input_layer/sparse_input_layer/input_layer/C11_embedding/C11_embedding_weights/embedding_lookup_sparse/embedding_lookup/axis*'
_output_shapes
:€€€€€€€€€*
Taxis0*
Tindices0	*
Tparams0*]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C11_embedding/embedding_weights
Є
Аinput_layer/sparse_input_layer/input_layer/C11_embedding/C11_embedding_weights/embedding_lookup_sparse/embedding_lookup/IdentityIdentitywinput_layer/sparse_input_layer/input_layer/C11_embedding/C11_embedding_weights/embedding_lookup_sparse/embedding_lookup*
T0*'
_output_shapes
:€€€€€€€€€
О
finput_layer/sparse_input_layer/input_layer/C11_embedding/C11_embedding_weights/embedding_lookup_sparseSparseSegmentMeanАinput_layer/sparse_input_layer/input_layer/C11_embedding/C11_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identityoinput_layer/sparse_input_layer/input_layer/C11_embedding/C11_embedding_weights/embedding_lookup_sparse/Unique:1kinput_layer/sparse_input_layer/input_layer/C11_embedding/C11_embedding_weights/embedding_lookup_sparse/Cast*'
_output_shapes
:€€€€€€€€€*
T0
ѓ
^input_layer/sparse_input_layer/input_layer/C11_embedding/C11_embedding_weights/Reshape_1/shapeConst*
valueB"€€€€   *
_output_shapes
:*
dtype0
п
Xinput_layer/sparse_input_layer/input_layer/C11_embedding/C11_embedding_weights/Reshape_1Reshapexinput_layer/sparse_input_layer/input_layer/C11_embedding/C11_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:2^input_layer/sparse_input_layer/input_layer/C11_embedding/C11_embedding_weights/Reshape_1/shape*
T0
*'
_output_shapes
:€€€€€€€€€
к
Tinput_layer/sparse_input_layer/input_layer/C11_embedding/C11_embedding_weights/ShapeShapefinput_layer/sparse_input_layer/input_layer/C11_embedding/C11_embedding_weights/embedding_lookup_sparse*
_output_shapes
:*
T0
ђ
binput_layer/sparse_input_layer/input_layer/C11_embedding/C11_embedding_weights/strided_slice/stackConst*
dtype0*
valueB:*
_output_shapes
:
Ѓ
dinput_layer/sparse_input_layer/input_layer/C11_embedding/C11_embedding_weights/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
Ѓ
dinput_layer/sparse_input_layer/input_layer/C11_embedding/C11_embedding_weights/strided_slice/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
Є
\input_layer/sparse_input_layer/input_layer/C11_embedding/C11_embedding_weights/strided_sliceStridedSliceTinput_layer/sparse_input_layer/input_layer/C11_embedding/C11_embedding_weights/Shapebinput_layer/sparse_input_layer/input_layer/C11_embedding/C11_embedding_weights/strided_slice/stackdinput_layer/sparse_input_layer/input_layer/C11_embedding/C11_embedding_weights/strided_slice/stack_1dinput_layer/sparse_input_layer/input_layer/C11_embedding/C11_embedding_weights/strided_slice/stack_2*
shrink_axis_mask*
_output_shapes
: *
T0*
Index0
Ш
Vinput_layer/sparse_input_layer/input_layer/C11_embedding/C11_embedding_weights/stack/0Const*
dtype0*
_output_shapes
: *
value	B :
ј
Tinput_layer/sparse_input_layer/input_layer/C11_embedding/C11_embedding_weights/stackPackVinput_layer/sparse_input_layer/input_layer/C11_embedding/C11_embedding_weights/stack/0\input_layer/sparse_input_layer/input_layer/C11_embedding/C11_embedding_weights/strided_slice*
N*
_output_shapes
:*
T0
∆
Sinput_layer/sparse_input_layer/input_layer/C11_embedding/C11_embedding_weights/TileTileXinput_layer/sparse_input_layer/input_layer/C11_embedding/C11_embedding_weights/Reshape_1Tinput_layer/sparse_input_layer/input_layer/C11_embedding/C11_embedding_weights/stack*
T0
*0
_output_shapes
:€€€€€€€€€€€€€€€€€€
А
Yinput_layer/sparse_input_layer/input_layer/C11_embedding/C11_embedding_weights/zeros_like	ZerosLikefinput_layer/sparse_input_layer/input_layer/C11_embedding/C11_embedding_weights/embedding_lookup_sparse*
T0*'
_output_shapes
:€€€€€€€€€
Ґ
Ninput_layer/sparse_input_layer/input_layer/C11_embedding/C11_embedding_weightsSelectSinput_layer/sparse_input_layer/input_layer/C11_embedding/C11_embedding_weights/TileYinput_layer/sparse_input_layer/input_layer/C11_embedding/C11_embedding_weights/zeros_likefinput_layer/sparse_input_layer/input_layer/C11_embedding/C11_embedding_weights/embedding_lookup_sparse*
T0*'
_output_shapes
:€€€€€€€€€
з
Uinput_layer/sparse_input_layer/input_layer/C11_embedding/C11_embedding_weights/Cast_1CastTinput_layer/sparse_input_layer/input_layer/C11_embedding/to_sparse_input/dense_shape*
_output_shapes
:*

SrcT0	*

DstT0
¶
\input_layer/sparse_input_layer/input_layer/C11_embedding/C11_embedding_weights/Slice_1/beginConst*
_output_shapes
:*
dtype0*
valueB: 
•
[input_layer/sparse_input_layer/input_layer/C11_embedding/C11_embedding_weights/Slice_1/sizeConst*
_output_shapes
:*
dtype0*
valueB:
£
Vinput_layer/sparse_input_layer/input_layer/C11_embedding/C11_embedding_weights/Slice_1SliceUinput_layer/sparse_input_layer/input_layer/C11_embedding/C11_embedding_weights/Cast_1\input_layer/sparse_input_layer/input_layer/C11_embedding/C11_embedding_weights/Slice_1/begin[input_layer/sparse_input_layer/input_layer/C11_embedding/C11_embedding_weights/Slice_1/size*
Index0*
_output_shapes
:*
T0
‘
Vinput_layer/sparse_input_layer/input_layer/C11_embedding/C11_embedding_weights/Shape_1ShapeNinput_layer/sparse_input_layer/input_layer/C11_embedding/C11_embedding_weights*
T0*
_output_shapes
:
¶
\input_layer/sparse_input_layer/input_layer/C11_embedding/C11_embedding_weights/Slice_2/beginConst*
_output_shapes
:*
valueB:*
dtype0
Ѓ
[input_layer/sparse_input_layer/input_layer/C11_embedding/C11_embedding_weights/Slice_2/sizeConst*
dtype0*
_output_shapes
:*
valueB:
€€€€€€€€€
§
Vinput_layer/sparse_input_layer/input_layer/C11_embedding/C11_embedding_weights/Slice_2SliceVinput_layer/sparse_input_layer/input_layer/C11_embedding/C11_embedding_weights/Shape_1\input_layer/sparse_input_layer/input_layer/C11_embedding/C11_embedding_weights/Slice_2/begin[input_layer/sparse_input_layer/input_layer/C11_embedding/C11_embedding_weights/Slice_2/size*
T0*
_output_shapes
:*
Index0
Ь
Zinput_layer/sparse_input_layer/input_layer/C11_embedding/C11_embedding_weights/concat/axisConst*
dtype0*
value	B : *
_output_shapes
: 
Ы
Uinput_layer/sparse_input_layer/input_layer/C11_embedding/C11_embedding_weights/concatConcatV2Vinput_layer/sparse_input_layer/input_layer/C11_embedding/C11_embedding_weights/Slice_1Vinput_layer/sparse_input_layer/input_layer/C11_embedding/C11_embedding_weights/Slice_2Zinput_layer/sparse_input_layer/input_layer/C11_embedding/C11_embedding_weights/concat/axis*
T0*
N*
_output_shapes
:
Љ
Xinput_layer/sparse_input_layer/input_layer/C11_embedding/C11_embedding_weights/Reshape_2ReshapeNinput_layer/sparse_input_layer/input_layer/C11_embedding/C11_embedding_weightsUinput_layer/sparse_input_layer/input_layer/C11_embedding/C11_embedding_weights/concat*'
_output_shapes
:€€€€€€€€€*
T0
∆
>input_layer/sparse_input_layer/input_layer/C11_embedding/ShapeShapeXinput_layer/sparse_input_layer/input_layer/C11_embedding/C11_embedding_weights/Reshape_2*
T0*
_output_shapes
:
Ц
Linput_layer/sparse_input_layer/input_layer/C11_embedding/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Ш
Ninput_layer/sparse_input_layer/input_layer/C11_embedding/strided_slice/stack_1Const*
_output_shapes
:*
valueB:*
dtype0
Ш
Ninput_layer/sparse_input_layer/input_layer/C11_embedding/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
 
Finput_layer/sparse_input_layer/input_layer/C11_embedding/strided_sliceStridedSlice>input_layer/sparse_input_layer/input_layer/C11_embedding/ShapeLinput_layer/sparse_input_layer/input_layer/C11_embedding/strided_slice/stackNinput_layer/sparse_input_layer/input_layer/C11_embedding/strided_slice/stack_1Ninput_layer/sparse_input_layer/input_layer/C11_embedding/strided_slice/stack_2*
Index0*
_output_shapes
: *
T0*
shrink_axis_mask
К
Hinput_layer/sparse_input_layer/input_layer/C11_embedding/Reshape/shape/1Const*
_output_shapes
: *
value	B :*
dtype0
О
Finput_layer/sparse_input_layer/input_layer/C11_embedding/Reshape/shapePackFinput_layer/sparse_input_layer/input_layer/C11_embedding/strided_sliceHinput_layer/sparse_input_layer/input_layer/C11_embedding/Reshape/shape/1*
_output_shapes
:*
T0*
N
Я
@input_layer/sparse_input_layer/input_layer/C11_embedding/ReshapeReshapeXinput_layer/sparse_input_layer/input_layer/C11_embedding/C11_embedding_weights/Reshape_2Finput_layer/sparse_input_layer/input_layer/C11_embedding/Reshape/shape*
T0*'
_output_shapes
:€€€€€€€€€
Т
Ginput_layer/sparse_input_layer/input_layer/C12_embedding/ExpandDims/dimConst*
valueB :
€€€€€€€€€*
_output_shapes
: *
dtype0
№
Cinput_layer/sparse_input_layer/input_layer/C12_embedding/ExpandDims
ExpandDimsPlaceholder_24Ginput_layer/sparse_input_layer/input_layer/C12_embedding/ExpandDims/dim*
T0*'
_output_shapes
:€€€€€€€€€
Ш
Winput_layer/sparse_input_layer/input_layer/C12_embedding/to_sparse_input/ignore_value/xConst*
dtype0*
valueB B *
_output_shapes
: 
≠
Qinput_layer/sparse_input_layer/input_layer/C12_embedding/to_sparse_input/NotEqualNotEqualCinput_layer/sparse_input_layer/input_layer/C12_embedding/ExpandDimsWinput_layer/sparse_input_layer/input_layer/C12_embedding/to_sparse_input/ignore_value/x*'
_output_shapes
:€€€€€€€€€*
T0
’
Pinput_layer/sparse_input_layer/input_layer/C12_embedding/to_sparse_input/indicesWhereQinput_layer/sparse_input_layer/input_layer/C12_embedding/to_sparse_input/NotEqual*'
_output_shapes
:€€€€€€€€€
ґ
Oinput_layer/sparse_input_layer/input_layer/C12_embedding/to_sparse_input/valuesGatherNdCinput_layer/sparse_input_layer/input_layer/C12_embedding/ExpandDimsPinput_layer/sparse_input_layer/input_layer/C12_embedding/to_sparse_input/indices*
Tparams0*
Tindices0	*#
_output_shapes
:€€€€€€€€€
„
Tinput_layer/sparse_input_layer/input_layer/C12_embedding/to_sparse_input/dense_shapeShapeCinput_layer/sparse_input_layer/input_layer/C12_embedding/ExpandDims*
out_type0	*
T0*
_output_shapes
:
г
?input_layer/sparse_input_layer/input_layer/C12_embedding/lookupStringToHashBucketFastOinput_layer/sparse_input_layer/input_layer/C12_embedding/to_sparse_input/values*#
_output_shapes
:€€€€€€€€€*
num_bucketsРN
Э
minput_layer/sparse_input_layer/input_layer/C12_embedding/embedding_weights/Initializer/truncated_normal/shapeConst*
_output_shapes
:*
valueB"'     *]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C12_embedding/embedding_weights*
dtype0
Р
linput_layer/sparse_input_layer/input_layer/C12_embedding/embedding_weights/Initializer/truncated_normal/meanConst*
dtype0*
_output_shapes
: *
valueB
 *    *]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C12_embedding/embedding_weights
Т
ninput_layer/sparse_input_layer/input_layer/C12_embedding/embedding_weights/Initializer/truncated_normal/stddevConst*
_output_shapes
: *
valueB
 *  А>*]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C12_embedding/embedding_weights*
dtype0
П
winput_layer/sparse_input_layer/input_layer/C12_embedding/embedding_weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormalminput_layer/sparse_input_layer/input_layer/C12_embedding/embedding_weights/Initializer/truncated_normal/shape*]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C12_embedding/embedding_weights*
_output_shapes
:	РN*
dtype0*
T0
д
kinput_layer/sparse_input_layer/input_layer/C12_embedding/embedding_weights/Initializer/truncated_normal/mulMulwinput_layer/sparse_input_layer/input_layer/C12_embedding/embedding_weights/Initializer/truncated_normal/TruncatedNormalninput_layer/sparse_input_layer/input_layer/C12_embedding/embedding_weights/Initializer/truncated_normal/stddev*
_output_shapes
:	РN*
T0*]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C12_embedding/embedding_weights
“
ginput_layer/sparse_input_layer/input_layer/C12_embedding/embedding_weights/Initializer/truncated_normalAddkinput_layer/sparse_input_layer/input_layer/C12_embedding/embedding_weights/Initializer/truncated_normal/mullinput_layer/sparse_input_layer/input_layer/C12_embedding/embedding_weights/Initializer/truncated_normal/mean*
_output_shapes
:	РN*
T0*]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C12_embedding/embedding_weights
ы
Jinput_layer/sparse_input_layer/input_layer/C12_embedding/embedding_weights
VariableV2*]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C12_embedding/embedding_weights*
dtype0*
shape:	РN*
_output_shapes
:	РN
Щ
Qinput_layer/sparse_input_layer/input_layer/C12_embedding/embedding_weights/AssignAssignJinput_layer/sparse_input_layer/input_layer/C12_embedding/embedding_weightsginput_layer/sparse_input_layer/input_layer/C12_embedding/embedding_weights/Initializer/truncated_normal*
T0*
_output_shapes
:	РN*]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C12_embedding/embedding_weights
∞
Oinput_layer/sparse_input_layer/input_layer/C12_embedding/embedding_weights/readIdentityJinput_layer/sparse_input_layer/input_layer/C12_embedding/embedding_weights*]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C12_embedding/embedding_weights*
T0*
_output_shapes
:	РN
§
Zinput_layer/sparse_input_layer/input_layer/C12_embedding/C12_embedding_weights/Slice/beginConst*
valueB: *
_output_shapes
:*
dtype0
£
Yinput_layer/sparse_input_layer/input_layer/C12_embedding/C12_embedding_weights/Slice/sizeConst*
valueB:*
dtype0*
_output_shapes
:
Ь
Tinput_layer/sparse_input_layer/input_layer/C12_embedding/C12_embedding_weights/SliceSliceTinput_layer/sparse_input_layer/input_layer/C12_embedding/to_sparse_input/dense_shapeZinput_layer/sparse_input_layer/input_layer/C12_embedding/C12_embedding_weights/Slice/beginYinput_layer/sparse_input_layer/input_layer/C12_embedding/C12_embedding_weights/Slice/size*
_output_shapes
:*
Index0*
T0	
Ю
Tinput_layer/sparse_input_layer/input_layer/C12_embedding/C12_embedding_weights/ConstConst*
_output_shapes
:*
valueB: *
dtype0
®
Sinput_layer/sparse_input_layer/input_layer/C12_embedding/C12_embedding_weights/ProdProdTinput_layer/sparse_input_layer/input_layer/C12_embedding/C12_embedding_weights/SliceTinput_layer/sparse_input_layer/input_layer/C12_embedding/C12_embedding_weights/Const*
_output_shapes
: *
T0	
°
_input_layer/sparse_input_layer/input_layer/C12_embedding/C12_embedding_weights/GatherV2/indicesConst*
value	B :*
dtype0*
_output_shapes
: 
Ю
\input_layer/sparse_input_layer/input_layer/C12_embedding/C12_embedding_weights/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 
Љ
Winput_layer/sparse_input_layer/input_layer/C12_embedding/C12_embedding_weights/GatherV2GatherV2Tinput_layer/sparse_input_layer/input_layer/C12_embedding/to_sparse_input/dense_shape_input_layer/sparse_input_layer/input_layer/C12_embedding/C12_embedding_weights/GatherV2/indices\input_layer/sparse_input_layer/input_layer/C12_embedding/C12_embedding_weights/GatherV2/axis*
Tindices0*
Taxis0*
_output_shapes
: *
Tparams0	
є
Uinput_layer/sparse_input_layer/input_layer/C12_embedding/C12_embedding_weights/Cast/xPackSinput_layer/sparse_input_layer/input_layer/C12_embedding/C12_embedding_weights/ProdWinput_layer/sparse_input_layer/input_layer/C12_embedding/C12_embedding_weights/GatherV2*
_output_shapes
:*
T0	*
N
Ы
\input_layer/sparse_input_layer/input_layer/C12_embedding/C12_embedding_weights/SparseReshapeSparseReshapePinput_layer/sparse_input_layer/input_layer/C12_embedding/to_sparse_input/indicesTinput_layer/sparse_input_layer/input_layer/C12_embedding/to_sparse_input/dense_shapeUinput_layer/sparse_input_layer/input_layer/C12_embedding/C12_embedding_weights/Cast/x*-
_output_shapes
:€€€€€€€€€:
а
einput_layer/sparse_input_layer/input_layer/C12_embedding/C12_embedding_weights/SparseReshape/IdentityIdentity?input_layer/sparse_input_layer/input_layer/C12_embedding/lookup*#
_output_shapes
:€€€€€€€€€*
T0	
Я
]input_layer/sparse_input_layer/input_layer/C12_embedding/C12_embedding_weights/GreaterEqual/yConst*
_output_shapes
: *
value	B	 R *
dtype0	
я
[input_layer/sparse_input_layer/input_layer/C12_embedding/C12_embedding_weights/GreaterEqualGreaterEqualeinput_layer/sparse_input_layer/input_layer/C12_embedding/C12_embedding_weights/SparseReshape/Identity]input_layer/sparse_input_layer/input_layer/C12_embedding/C12_embedding_weights/GreaterEqual/y*
T0	*#
_output_shapes
:€€€€€€€€€
г
Tinput_layer/sparse_input_layer/input_layer/C12_embedding/C12_embedding_weights/WhereWhere[input_layer/sparse_input_layer/input_layer/C12_embedding/C12_embedding_weights/GreaterEqual*'
_output_shapes
:€€€€€€€€€
ѓ
\input_layer/sparse_input_layer/input_layer/C12_embedding/C12_embedding_weights/Reshape/shapeConst*
dtype0*
valueB:
€€€€€€€€€*
_output_shapes
:
√
Vinput_layer/sparse_input_layer/input_layer/C12_embedding/C12_embedding_weights/ReshapeReshapeTinput_layer/sparse_input_layer/input_layer/C12_embedding/C12_embedding_weights/Where\input_layer/sparse_input_layer/input_layer/C12_embedding/C12_embedding_weights/Reshape/shape*
T0	*#
_output_shapes
:€€€€€€€€€
†
^input_layer/sparse_input_layer/input_layer/C12_embedding/C12_embedding_weights/GatherV2_1/axisConst*
dtype0*
_output_shapes
: *
value	B : 
–
Yinput_layer/sparse_input_layer/input_layer/C12_embedding/C12_embedding_weights/GatherV2_1GatherV2\input_layer/sparse_input_layer/input_layer/C12_embedding/C12_embedding_weights/SparseReshapeVinput_layer/sparse_input_layer/input_layer/C12_embedding/C12_embedding_weights/Reshape^input_layer/sparse_input_layer/input_layer/C12_embedding/C12_embedding_weights/GatherV2_1/axis*
Tparams0	*
Tindices0	*
Taxis0*'
_output_shapes
:€€€€€€€€€
†
^input_layer/sparse_input_layer/input_layer/C12_embedding/C12_embedding_weights/GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
value	B : 
’
Yinput_layer/sparse_input_layer/input_layer/C12_embedding/C12_embedding_weights/GatherV2_2GatherV2einput_layer/sparse_input_layer/input_layer/C12_embedding/C12_embedding_weights/SparseReshape/IdentityVinput_layer/sparse_input_layer/input_layer/C12_embedding/C12_embedding_weights/Reshape^input_layer/sparse_input_layer/input_layer/C12_embedding/C12_embedding_weights/GatherV2_2/axis*
Taxis0*
Tparams0	*#
_output_shapes
:€€€€€€€€€*
Tindices0	
и
Winput_layer/sparse_input_layer/input_layer/C12_embedding/C12_embedding_weights/IdentityIdentity^input_layer/sparse_input_layer/input_layer/C12_embedding/C12_embedding_weights/SparseReshape:1*
T0	*
_output_shapes
:
™
hinput_layer/sparse_input_layer/input_layer/C12_embedding/C12_embedding_weights/SparseFillEmptyRows/ConstConst*
dtype0	*
_output_shapes
: *
value	B	 R 
е
vinput_layer/sparse_input_layer/input_layer/C12_embedding/C12_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowsSparseFillEmptyRowsYinput_layer/sparse_input_layer/input_layer/C12_embedding/C12_embedding_weights/GatherV2_1Yinput_layer/sparse_input_layer/input_layer/C12_embedding/C12_embedding_weights/GatherV2_2Winput_layer/sparse_input_layer/input_layer/C12_embedding/C12_embedding_weights/Identityhinput_layer/sparse_input_layer/input_layer/C12_embedding/C12_embedding_weights/SparseFillEmptyRows/Const*T
_output_shapesB
@:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€*
T0	
Ћ
zinput_layer/sparse_input_layer/input_layer/C12_embedding/C12_embedding_weights/embedding_lookup_sparse/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB"        
Ќ
|input_layer/sparse_input_layer/input_layer/C12_embedding/C12_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1Const*
_output_shapes
:*
valueB"       *
dtype0
Ќ
|input_layer/sparse_input_layer/input_layer/C12_embedding/C12_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2Const*
valueB"      *
_output_shapes
:*
dtype0
й
tinput_layer/sparse_input_layer/input_layer/C12_embedding/C12_embedding_weights/embedding_lookup_sparse/strided_sliceStridedSlicevinput_layer/sparse_input_layer/input_layer/C12_embedding/C12_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowszinput_layer/sparse_input_layer/input_layer/C12_embedding/C12_embedding_weights/embedding_lookup_sparse/strided_slice/stack|input_layer/sparse_input_layer/input_layer/C12_embedding/C12_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1|input_layer/sparse_input_layer/input_layer/C12_embedding/C12_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2*
T0	*#
_output_shapes
:€€€€€€€€€*

begin_mask*
end_mask*
shrink_axis_mask*
Index0
¶
kinput_layer/sparse_input_layer/input_layer/C12_embedding/C12_embedding_weights/embedding_lookup_sparse/CastCasttinput_layer/sparse_input_layer/input_layer/C12_embedding/C12_embedding_weights/embedding_lookup_sparse/strided_slice*

SrcT0	*#
_output_shapes
:€€€€€€€€€*

DstT0
Ѓ
minput_layer/sparse_input_layer/input_layer/C12_embedding/C12_embedding_weights/embedding_lookup_sparse/UniqueUniquexinput_layer/sparse_input_layer/input_layer/C12_embedding/C12_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:1*
T0	*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
Э
|input_layer/sparse_input_layer/input_layer/C12_embedding/C12_embedding_weights/embedding_lookup_sparse/embedding_lookup/axisConst*
dtype0*
_output_shapes
: *]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C12_embedding/embedding_weights*
value	B : 
х
winput_layer/sparse_input_layer/input_layer/C12_embedding/C12_embedding_weights/embedding_lookup_sparse/embedding_lookupGatherV2Oinput_layer/sparse_input_layer/input_layer/C12_embedding/embedding_weights/readminput_layer/sparse_input_layer/input_layer/C12_embedding/C12_embedding_weights/embedding_lookup_sparse/Unique|input_layer/sparse_input_layer/input_layer/C12_embedding/C12_embedding_weights/embedding_lookup_sparse/embedding_lookup/axis*'
_output_shapes
:€€€€€€€€€*
Tindices0	*
Tparams0*
Taxis0*]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C12_embedding/embedding_weights
Є
Аinput_layer/sparse_input_layer/input_layer/C12_embedding/C12_embedding_weights/embedding_lookup_sparse/embedding_lookup/IdentityIdentitywinput_layer/sparse_input_layer/input_layer/C12_embedding/C12_embedding_weights/embedding_lookup_sparse/embedding_lookup*
T0*'
_output_shapes
:€€€€€€€€€
О
finput_layer/sparse_input_layer/input_layer/C12_embedding/C12_embedding_weights/embedding_lookup_sparseSparseSegmentMeanАinput_layer/sparse_input_layer/input_layer/C12_embedding/C12_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identityoinput_layer/sparse_input_layer/input_layer/C12_embedding/C12_embedding_weights/embedding_lookup_sparse/Unique:1kinput_layer/sparse_input_layer/input_layer/C12_embedding/C12_embedding_weights/embedding_lookup_sparse/Cast*
T0*'
_output_shapes
:€€€€€€€€€
ѓ
^input_layer/sparse_input_layer/input_layer/C12_embedding/C12_embedding_weights/Reshape_1/shapeConst*
dtype0*
valueB"€€€€   *
_output_shapes
:
п
Xinput_layer/sparse_input_layer/input_layer/C12_embedding/C12_embedding_weights/Reshape_1Reshapexinput_layer/sparse_input_layer/input_layer/C12_embedding/C12_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:2^input_layer/sparse_input_layer/input_layer/C12_embedding/C12_embedding_weights/Reshape_1/shape*'
_output_shapes
:€€€€€€€€€*
T0

к
Tinput_layer/sparse_input_layer/input_layer/C12_embedding/C12_embedding_weights/ShapeShapefinput_layer/sparse_input_layer/input_layer/C12_embedding/C12_embedding_weights/embedding_lookup_sparse*
T0*
_output_shapes
:
ђ
binput_layer/sparse_input_layer/input_layer/C12_embedding/C12_embedding_weights/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB:
Ѓ
dinput_layer/sparse_input_layer/input_layer/C12_embedding/C12_embedding_weights/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
Ѓ
dinput_layer/sparse_input_layer/input_layer/C12_embedding/C12_embedding_weights/strided_slice/stack_2Const*
valueB:*
_output_shapes
:*
dtype0
Є
\input_layer/sparse_input_layer/input_layer/C12_embedding/C12_embedding_weights/strided_sliceStridedSliceTinput_layer/sparse_input_layer/input_layer/C12_embedding/C12_embedding_weights/Shapebinput_layer/sparse_input_layer/input_layer/C12_embedding/C12_embedding_weights/strided_slice/stackdinput_layer/sparse_input_layer/input_layer/C12_embedding/C12_embedding_weights/strided_slice/stack_1dinput_layer/sparse_input_layer/input_layer/C12_embedding/C12_embedding_weights/strided_slice/stack_2*
T0*
_output_shapes
: *
Index0*
shrink_axis_mask
Ш
Vinput_layer/sparse_input_layer/input_layer/C12_embedding/C12_embedding_weights/stack/0Const*
_output_shapes
: *
dtype0*
value	B :
ј
Tinput_layer/sparse_input_layer/input_layer/C12_embedding/C12_embedding_weights/stackPackVinput_layer/sparse_input_layer/input_layer/C12_embedding/C12_embedding_weights/stack/0\input_layer/sparse_input_layer/input_layer/C12_embedding/C12_embedding_weights/strided_slice*
_output_shapes
:*
T0*
N
∆
Sinput_layer/sparse_input_layer/input_layer/C12_embedding/C12_embedding_weights/TileTileXinput_layer/sparse_input_layer/input_layer/C12_embedding/C12_embedding_weights/Reshape_1Tinput_layer/sparse_input_layer/input_layer/C12_embedding/C12_embedding_weights/stack*
T0
*0
_output_shapes
:€€€€€€€€€€€€€€€€€€
А
Yinput_layer/sparse_input_layer/input_layer/C12_embedding/C12_embedding_weights/zeros_like	ZerosLikefinput_layer/sparse_input_layer/input_layer/C12_embedding/C12_embedding_weights/embedding_lookup_sparse*'
_output_shapes
:€€€€€€€€€*
T0
Ґ
Ninput_layer/sparse_input_layer/input_layer/C12_embedding/C12_embedding_weightsSelectSinput_layer/sparse_input_layer/input_layer/C12_embedding/C12_embedding_weights/TileYinput_layer/sparse_input_layer/input_layer/C12_embedding/C12_embedding_weights/zeros_likefinput_layer/sparse_input_layer/input_layer/C12_embedding/C12_embedding_weights/embedding_lookup_sparse*'
_output_shapes
:€€€€€€€€€*
T0
з
Uinput_layer/sparse_input_layer/input_layer/C12_embedding/C12_embedding_weights/Cast_1CastTinput_layer/sparse_input_layer/input_layer/C12_embedding/to_sparse_input/dense_shape*
_output_shapes
:*

DstT0*

SrcT0	
¶
\input_layer/sparse_input_layer/input_layer/C12_embedding/C12_embedding_weights/Slice_1/beginConst*
valueB: *
_output_shapes
:*
dtype0
•
[input_layer/sparse_input_layer/input_layer/C12_embedding/C12_embedding_weights/Slice_1/sizeConst*
dtype0*
valueB:*
_output_shapes
:
£
Vinput_layer/sparse_input_layer/input_layer/C12_embedding/C12_embedding_weights/Slice_1SliceUinput_layer/sparse_input_layer/input_layer/C12_embedding/C12_embedding_weights/Cast_1\input_layer/sparse_input_layer/input_layer/C12_embedding/C12_embedding_weights/Slice_1/begin[input_layer/sparse_input_layer/input_layer/C12_embedding/C12_embedding_weights/Slice_1/size*
_output_shapes
:*
Index0*
T0
‘
Vinput_layer/sparse_input_layer/input_layer/C12_embedding/C12_embedding_weights/Shape_1ShapeNinput_layer/sparse_input_layer/input_layer/C12_embedding/C12_embedding_weights*
T0*
_output_shapes
:
¶
\input_layer/sparse_input_layer/input_layer/C12_embedding/C12_embedding_weights/Slice_2/beginConst*
dtype0*
_output_shapes
:*
valueB:
Ѓ
[input_layer/sparse_input_layer/input_layer/C12_embedding/C12_embedding_weights/Slice_2/sizeConst*
valueB:
€€€€€€€€€*
dtype0*
_output_shapes
:
§
Vinput_layer/sparse_input_layer/input_layer/C12_embedding/C12_embedding_weights/Slice_2SliceVinput_layer/sparse_input_layer/input_layer/C12_embedding/C12_embedding_weights/Shape_1\input_layer/sparse_input_layer/input_layer/C12_embedding/C12_embedding_weights/Slice_2/begin[input_layer/sparse_input_layer/input_layer/C12_embedding/C12_embedding_weights/Slice_2/size*
T0*
Index0*
_output_shapes
:
Ь
Zinput_layer/sparse_input_layer/input_layer/C12_embedding/C12_embedding_weights/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
Ы
Uinput_layer/sparse_input_layer/input_layer/C12_embedding/C12_embedding_weights/concatConcatV2Vinput_layer/sparse_input_layer/input_layer/C12_embedding/C12_embedding_weights/Slice_1Vinput_layer/sparse_input_layer/input_layer/C12_embedding/C12_embedding_weights/Slice_2Zinput_layer/sparse_input_layer/input_layer/C12_embedding/C12_embedding_weights/concat/axis*
T0*
N*
_output_shapes
:
Љ
Xinput_layer/sparse_input_layer/input_layer/C12_embedding/C12_embedding_weights/Reshape_2ReshapeNinput_layer/sparse_input_layer/input_layer/C12_embedding/C12_embedding_weightsUinput_layer/sparse_input_layer/input_layer/C12_embedding/C12_embedding_weights/concat*'
_output_shapes
:€€€€€€€€€*
T0
∆
>input_layer/sparse_input_layer/input_layer/C12_embedding/ShapeShapeXinput_layer/sparse_input_layer/input_layer/C12_embedding/C12_embedding_weights/Reshape_2*
_output_shapes
:*
T0
Ц
Linput_layer/sparse_input_layer/input_layer/C12_embedding/strided_slice/stackConst*
dtype0*
valueB: *
_output_shapes
:
Ш
Ninput_layer/sparse_input_layer/input_layer/C12_embedding/strided_slice/stack_1Const*
_output_shapes
:*
valueB:*
dtype0
Ш
Ninput_layer/sparse_input_layer/input_layer/C12_embedding/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
 
Finput_layer/sparse_input_layer/input_layer/C12_embedding/strided_sliceStridedSlice>input_layer/sparse_input_layer/input_layer/C12_embedding/ShapeLinput_layer/sparse_input_layer/input_layer/C12_embedding/strided_slice/stackNinput_layer/sparse_input_layer/input_layer/C12_embedding/strided_slice/stack_1Ninput_layer/sparse_input_layer/input_layer/C12_embedding/strided_slice/stack_2*
shrink_axis_mask*
Index0*
_output_shapes
: *
T0
К
Hinput_layer/sparse_input_layer/input_layer/C12_embedding/Reshape/shape/1Const*
_output_shapes
: *
value	B :*
dtype0
О
Finput_layer/sparse_input_layer/input_layer/C12_embedding/Reshape/shapePackFinput_layer/sparse_input_layer/input_layer/C12_embedding/strided_sliceHinput_layer/sparse_input_layer/input_layer/C12_embedding/Reshape/shape/1*
N*
T0*
_output_shapes
:
Я
@input_layer/sparse_input_layer/input_layer/C12_embedding/ReshapeReshapeXinput_layer/sparse_input_layer/input_layer/C12_embedding/C12_embedding_weights/Reshape_2Finput_layer/sparse_input_layer/input_layer/C12_embedding/Reshape/shape*'
_output_shapes
:€€€€€€€€€*
T0
Т
Ginput_layer/sparse_input_layer/input_layer/C13_embedding/ExpandDims/dimConst*
dtype0*
valueB :
€€€€€€€€€*
_output_shapes
: 
№
Cinput_layer/sparse_input_layer/input_layer/C13_embedding/ExpandDims
ExpandDimsPlaceholder_25Ginput_layer/sparse_input_layer/input_layer/C13_embedding/ExpandDims/dim*'
_output_shapes
:€€€€€€€€€*
T0
Ш
Winput_layer/sparse_input_layer/input_layer/C13_embedding/to_sparse_input/ignore_value/xConst*
dtype0*
valueB B *
_output_shapes
: 
≠
Qinput_layer/sparse_input_layer/input_layer/C13_embedding/to_sparse_input/NotEqualNotEqualCinput_layer/sparse_input_layer/input_layer/C13_embedding/ExpandDimsWinput_layer/sparse_input_layer/input_layer/C13_embedding/to_sparse_input/ignore_value/x*
T0*'
_output_shapes
:€€€€€€€€€
’
Pinput_layer/sparse_input_layer/input_layer/C13_embedding/to_sparse_input/indicesWhereQinput_layer/sparse_input_layer/input_layer/C13_embedding/to_sparse_input/NotEqual*'
_output_shapes
:€€€€€€€€€
ґ
Oinput_layer/sparse_input_layer/input_layer/C13_embedding/to_sparse_input/valuesGatherNdCinput_layer/sparse_input_layer/input_layer/C13_embedding/ExpandDimsPinput_layer/sparse_input_layer/input_layer/C13_embedding/to_sparse_input/indices*
Tparams0*
Tindices0	*#
_output_shapes
:€€€€€€€€€
„
Tinput_layer/sparse_input_layer/input_layer/C13_embedding/to_sparse_input/dense_shapeShapeCinput_layer/sparse_input_layer/input_layer/C13_embedding/ExpandDims*
out_type0	*
_output_shapes
:*
T0
г
?input_layer/sparse_input_layer/input_layer/C13_embedding/lookupStringToHashBucketFastOinput_layer/sparse_input_layer/input_layer/C13_embedding/to_sparse_input/values*#
_output_shapes
:€€€€€€€€€*
num_bucketsРN
Э
minput_layer/sparse_input_layer/input_layer/C13_embedding/embedding_weights/Initializer/truncated_normal/shapeConst*]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C13_embedding/embedding_weights*
valueB"'     *
dtype0*
_output_shapes
:
Р
linput_layer/sparse_input_layer/input_layer/C13_embedding/embedding_weights/Initializer/truncated_normal/meanConst*]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C13_embedding/embedding_weights*
dtype0*
valueB
 *    *
_output_shapes
: 
Т
ninput_layer/sparse_input_layer/input_layer/C13_embedding/embedding_weights/Initializer/truncated_normal/stddevConst*
valueB
 *  А>*]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C13_embedding/embedding_weights*
_output_shapes
: *
dtype0
П
winput_layer/sparse_input_layer/input_layer/C13_embedding/embedding_weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormalminput_layer/sparse_input_layer/input_layer/C13_embedding/embedding_weights/Initializer/truncated_normal/shape*]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C13_embedding/embedding_weights*
_output_shapes
:	РN*
T0*
dtype0
д
kinput_layer/sparse_input_layer/input_layer/C13_embedding/embedding_weights/Initializer/truncated_normal/mulMulwinput_layer/sparse_input_layer/input_layer/C13_embedding/embedding_weights/Initializer/truncated_normal/TruncatedNormalninput_layer/sparse_input_layer/input_layer/C13_embedding/embedding_weights/Initializer/truncated_normal/stddev*
T0*
_output_shapes
:	РN*]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C13_embedding/embedding_weights
“
ginput_layer/sparse_input_layer/input_layer/C13_embedding/embedding_weights/Initializer/truncated_normalAddkinput_layer/sparse_input_layer/input_layer/C13_embedding/embedding_weights/Initializer/truncated_normal/mullinput_layer/sparse_input_layer/input_layer/C13_embedding/embedding_weights/Initializer/truncated_normal/mean*
_output_shapes
:	РN*]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C13_embedding/embedding_weights*
T0
ы
Jinput_layer/sparse_input_layer/input_layer/C13_embedding/embedding_weights
VariableV2*
shape:	РN*]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C13_embedding/embedding_weights*
_output_shapes
:	РN*
dtype0
Щ
Qinput_layer/sparse_input_layer/input_layer/C13_embedding/embedding_weights/AssignAssignJinput_layer/sparse_input_layer/input_layer/C13_embedding/embedding_weightsginput_layer/sparse_input_layer/input_layer/C13_embedding/embedding_weights/Initializer/truncated_normal*
T0*]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C13_embedding/embedding_weights*
_output_shapes
:	РN
∞
Oinput_layer/sparse_input_layer/input_layer/C13_embedding/embedding_weights/readIdentityJinput_layer/sparse_input_layer/input_layer/C13_embedding/embedding_weights*]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C13_embedding/embedding_weights*
_output_shapes
:	РN*
T0
§
Zinput_layer/sparse_input_layer/input_layer/C13_embedding/C13_embedding_weights/Slice/beginConst*
dtype0*
valueB: *
_output_shapes
:
£
Yinput_layer/sparse_input_layer/input_layer/C13_embedding/C13_embedding_weights/Slice/sizeConst*
dtype0*
_output_shapes
:*
valueB:
Ь
Tinput_layer/sparse_input_layer/input_layer/C13_embedding/C13_embedding_weights/SliceSliceTinput_layer/sparse_input_layer/input_layer/C13_embedding/to_sparse_input/dense_shapeZinput_layer/sparse_input_layer/input_layer/C13_embedding/C13_embedding_weights/Slice/beginYinput_layer/sparse_input_layer/input_layer/C13_embedding/C13_embedding_weights/Slice/size*
Index0*
T0	*
_output_shapes
:
Ю
Tinput_layer/sparse_input_layer/input_layer/C13_embedding/C13_embedding_weights/ConstConst*
_output_shapes
:*
valueB: *
dtype0
®
Sinput_layer/sparse_input_layer/input_layer/C13_embedding/C13_embedding_weights/ProdProdTinput_layer/sparse_input_layer/input_layer/C13_embedding/C13_embedding_weights/SliceTinput_layer/sparse_input_layer/input_layer/C13_embedding/C13_embedding_weights/Const*
T0	*
_output_shapes
: 
°
_input_layer/sparse_input_layer/input_layer/C13_embedding/C13_embedding_weights/GatherV2/indicesConst*
dtype0*
_output_shapes
: *
value	B :
Ю
\input_layer/sparse_input_layer/input_layer/C13_embedding/C13_embedding_weights/GatherV2/axisConst*
dtype0*
_output_shapes
: *
value	B : 
Љ
Winput_layer/sparse_input_layer/input_layer/C13_embedding/C13_embedding_weights/GatherV2GatherV2Tinput_layer/sparse_input_layer/input_layer/C13_embedding/to_sparse_input/dense_shape_input_layer/sparse_input_layer/input_layer/C13_embedding/C13_embedding_weights/GatherV2/indices\input_layer/sparse_input_layer/input_layer/C13_embedding/C13_embedding_weights/GatherV2/axis*
Tindices0*
Taxis0*
Tparams0	*
_output_shapes
: 
є
Uinput_layer/sparse_input_layer/input_layer/C13_embedding/C13_embedding_weights/Cast/xPackSinput_layer/sparse_input_layer/input_layer/C13_embedding/C13_embedding_weights/ProdWinput_layer/sparse_input_layer/input_layer/C13_embedding/C13_embedding_weights/GatherV2*
_output_shapes
:*
T0	*
N
Ы
\input_layer/sparse_input_layer/input_layer/C13_embedding/C13_embedding_weights/SparseReshapeSparseReshapePinput_layer/sparse_input_layer/input_layer/C13_embedding/to_sparse_input/indicesTinput_layer/sparse_input_layer/input_layer/C13_embedding/to_sparse_input/dense_shapeUinput_layer/sparse_input_layer/input_layer/C13_embedding/C13_embedding_weights/Cast/x*-
_output_shapes
:€€€€€€€€€:
а
einput_layer/sparse_input_layer/input_layer/C13_embedding/C13_embedding_weights/SparseReshape/IdentityIdentity?input_layer/sparse_input_layer/input_layer/C13_embedding/lookup*#
_output_shapes
:€€€€€€€€€*
T0	
Я
]input_layer/sparse_input_layer/input_layer/C13_embedding/C13_embedding_weights/GreaterEqual/yConst*
value	B	 R *
_output_shapes
: *
dtype0	
я
[input_layer/sparse_input_layer/input_layer/C13_embedding/C13_embedding_weights/GreaterEqualGreaterEqualeinput_layer/sparse_input_layer/input_layer/C13_embedding/C13_embedding_weights/SparseReshape/Identity]input_layer/sparse_input_layer/input_layer/C13_embedding/C13_embedding_weights/GreaterEqual/y*#
_output_shapes
:€€€€€€€€€*
T0	
г
Tinput_layer/sparse_input_layer/input_layer/C13_embedding/C13_embedding_weights/WhereWhere[input_layer/sparse_input_layer/input_layer/C13_embedding/C13_embedding_weights/GreaterEqual*'
_output_shapes
:€€€€€€€€€
ѓ
\input_layer/sparse_input_layer/input_layer/C13_embedding/C13_embedding_weights/Reshape/shapeConst*
valueB:
€€€€€€€€€*
dtype0*
_output_shapes
:
√
Vinput_layer/sparse_input_layer/input_layer/C13_embedding/C13_embedding_weights/ReshapeReshapeTinput_layer/sparse_input_layer/input_layer/C13_embedding/C13_embedding_weights/Where\input_layer/sparse_input_layer/input_layer/C13_embedding/C13_embedding_weights/Reshape/shape*#
_output_shapes
:€€€€€€€€€*
T0	
†
^input_layer/sparse_input_layer/input_layer/C13_embedding/C13_embedding_weights/GatherV2_1/axisConst*
dtype0*
value	B : *
_output_shapes
: 
–
Yinput_layer/sparse_input_layer/input_layer/C13_embedding/C13_embedding_weights/GatherV2_1GatherV2\input_layer/sparse_input_layer/input_layer/C13_embedding/C13_embedding_weights/SparseReshapeVinput_layer/sparse_input_layer/input_layer/C13_embedding/C13_embedding_weights/Reshape^input_layer/sparse_input_layer/input_layer/C13_embedding/C13_embedding_weights/GatherV2_1/axis*
Tparams0	*
Tindices0	*'
_output_shapes
:€€€€€€€€€*
Taxis0
†
^input_layer/sparse_input_layer/input_layer/C13_embedding/C13_embedding_weights/GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
value	B : 
’
Yinput_layer/sparse_input_layer/input_layer/C13_embedding/C13_embedding_weights/GatherV2_2GatherV2einput_layer/sparse_input_layer/input_layer/C13_embedding/C13_embedding_weights/SparseReshape/IdentityVinput_layer/sparse_input_layer/input_layer/C13_embedding/C13_embedding_weights/Reshape^input_layer/sparse_input_layer/input_layer/C13_embedding/C13_embedding_weights/GatherV2_2/axis*
Taxis0*
Tindices0	*#
_output_shapes
:€€€€€€€€€*
Tparams0	
и
Winput_layer/sparse_input_layer/input_layer/C13_embedding/C13_embedding_weights/IdentityIdentity^input_layer/sparse_input_layer/input_layer/C13_embedding/C13_embedding_weights/SparseReshape:1*
_output_shapes
:*
T0	
™
hinput_layer/sparse_input_layer/input_layer/C13_embedding/C13_embedding_weights/SparseFillEmptyRows/ConstConst*
dtype0	*
value	B	 R *
_output_shapes
: 
е
vinput_layer/sparse_input_layer/input_layer/C13_embedding/C13_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowsSparseFillEmptyRowsYinput_layer/sparse_input_layer/input_layer/C13_embedding/C13_embedding_weights/GatherV2_1Yinput_layer/sparse_input_layer/input_layer/C13_embedding/C13_embedding_weights/GatherV2_2Winput_layer/sparse_input_layer/input_layer/C13_embedding/C13_embedding_weights/Identityhinput_layer/sparse_input_layer/input_layer/C13_embedding/C13_embedding_weights/SparseFillEmptyRows/Const*
T0	*T
_output_shapesB
@:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€
Ћ
zinput_layer/sparse_input_layer/input_layer/C13_embedding/C13_embedding_weights/embedding_lookup_sparse/strided_slice/stackConst*
dtype0*
valueB"        *
_output_shapes
:
Ќ
|input_layer/sparse_input_layer/input_layer/C13_embedding/C13_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1Const*
valueB"       *
dtype0*
_output_shapes
:
Ќ
|input_layer/sparse_input_layer/input_layer/C13_embedding/C13_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:
й
tinput_layer/sparse_input_layer/input_layer/C13_embedding/C13_embedding_weights/embedding_lookup_sparse/strided_sliceStridedSlicevinput_layer/sparse_input_layer/input_layer/C13_embedding/C13_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowszinput_layer/sparse_input_layer/input_layer/C13_embedding/C13_embedding_weights/embedding_lookup_sparse/strided_slice/stack|input_layer/sparse_input_layer/input_layer/C13_embedding/C13_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1|input_layer/sparse_input_layer/input_layer/C13_embedding/C13_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2*
end_mask*
Index0*
T0	*#
_output_shapes
:€€€€€€€€€*

begin_mask*
shrink_axis_mask
¶
kinput_layer/sparse_input_layer/input_layer/C13_embedding/C13_embedding_weights/embedding_lookup_sparse/CastCasttinput_layer/sparse_input_layer/input_layer/C13_embedding/C13_embedding_weights/embedding_lookup_sparse/strided_slice*

SrcT0	*

DstT0*#
_output_shapes
:€€€€€€€€€
Ѓ
minput_layer/sparse_input_layer/input_layer/C13_embedding/C13_embedding_weights/embedding_lookup_sparse/UniqueUniquexinput_layer/sparse_input_layer/input_layer/C13_embedding/C13_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€*
T0	
Э
|input_layer/sparse_input_layer/input_layer/C13_embedding/C13_embedding_weights/embedding_lookup_sparse/embedding_lookup/axisConst*
dtype0*]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C13_embedding/embedding_weights*
value	B : *
_output_shapes
: 
х
winput_layer/sparse_input_layer/input_layer/C13_embedding/C13_embedding_weights/embedding_lookup_sparse/embedding_lookupGatherV2Oinput_layer/sparse_input_layer/input_layer/C13_embedding/embedding_weights/readminput_layer/sparse_input_layer/input_layer/C13_embedding/C13_embedding_weights/embedding_lookup_sparse/Unique|input_layer/sparse_input_layer/input_layer/C13_embedding/C13_embedding_weights/embedding_lookup_sparse/embedding_lookup/axis*
Tparams0*
Tindices0	*'
_output_shapes
:€€€€€€€€€*
Taxis0*]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C13_embedding/embedding_weights
Є
Аinput_layer/sparse_input_layer/input_layer/C13_embedding/C13_embedding_weights/embedding_lookup_sparse/embedding_lookup/IdentityIdentitywinput_layer/sparse_input_layer/input_layer/C13_embedding/C13_embedding_weights/embedding_lookup_sparse/embedding_lookup*
T0*'
_output_shapes
:€€€€€€€€€
О
finput_layer/sparse_input_layer/input_layer/C13_embedding/C13_embedding_weights/embedding_lookup_sparseSparseSegmentMeanАinput_layer/sparse_input_layer/input_layer/C13_embedding/C13_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identityoinput_layer/sparse_input_layer/input_layer/C13_embedding/C13_embedding_weights/embedding_lookup_sparse/Unique:1kinput_layer/sparse_input_layer/input_layer/C13_embedding/C13_embedding_weights/embedding_lookup_sparse/Cast*
T0*'
_output_shapes
:€€€€€€€€€
ѓ
^input_layer/sparse_input_layer/input_layer/C13_embedding/C13_embedding_weights/Reshape_1/shapeConst*
valueB"€€€€   *
dtype0*
_output_shapes
:
п
Xinput_layer/sparse_input_layer/input_layer/C13_embedding/C13_embedding_weights/Reshape_1Reshapexinput_layer/sparse_input_layer/input_layer/C13_embedding/C13_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:2^input_layer/sparse_input_layer/input_layer/C13_embedding/C13_embedding_weights/Reshape_1/shape*
T0
*'
_output_shapes
:€€€€€€€€€
к
Tinput_layer/sparse_input_layer/input_layer/C13_embedding/C13_embedding_weights/ShapeShapefinput_layer/sparse_input_layer/input_layer/C13_embedding/C13_embedding_weights/embedding_lookup_sparse*
_output_shapes
:*
T0
ђ
binput_layer/sparse_input_layer/input_layer/C13_embedding/C13_embedding_weights/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
Ѓ
dinput_layer/sparse_input_layer/input_layer/C13_embedding/C13_embedding_weights/strided_slice/stack_1Const*
dtype0*
valueB:*
_output_shapes
:
Ѓ
dinput_layer/sparse_input_layer/input_layer/C13_embedding/C13_embedding_weights/strided_slice/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
Є
\input_layer/sparse_input_layer/input_layer/C13_embedding/C13_embedding_weights/strided_sliceStridedSliceTinput_layer/sparse_input_layer/input_layer/C13_embedding/C13_embedding_weights/Shapebinput_layer/sparse_input_layer/input_layer/C13_embedding/C13_embedding_weights/strided_slice/stackdinput_layer/sparse_input_layer/input_layer/C13_embedding/C13_embedding_weights/strided_slice/stack_1dinput_layer/sparse_input_layer/input_layer/C13_embedding/C13_embedding_weights/strided_slice/stack_2*
T0*
_output_shapes
: *
Index0*
shrink_axis_mask
Ш
Vinput_layer/sparse_input_layer/input_layer/C13_embedding/C13_embedding_weights/stack/0Const*
value	B :*
dtype0*
_output_shapes
: 
ј
Tinput_layer/sparse_input_layer/input_layer/C13_embedding/C13_embedding_weights/stackPackVinput_layer/sparse_input_layer/input_layer/C13_embedding/C13_embedding_weights/stack/0\input_layer/sparse_input_layer/input_layer/C13_embedding/C13_embedding_weights/strided_slice*
_output_shapes
:*
N*
T0
∆
Sinput_layer/sparse_input_layer/input_layer/C13_embedding/C13_embedding_weights/TileTileXinput_layer/sparse_input_layer/input_layer/C13_embedding/C13_embedding_weights/Reshape_1Tinput_layer/sparse_input_layer/input_layer/C13_embedding/C13_embedding_weights/stack*
T0
*0
_output_shapes
:€€€€€€€€€€€€€€€€€€
А
Yinput_layer/sparse_input_layer/input_layer/C13_embedding/C13_embedding_weights/zeros_like	ZerosLikefinput_layer/sparse_input_layer/input_layer/C13_embedding/C13_embedding_weights/embedding_lookup_sparse*
T0*'
_output_shapes
:€€€€€€€€€
Ґ
Ninput_layer/sparse_input_layer/input_layer/C13_embedding/C13_embedding_weightsSelectSinput_layer/sparse_input_layer/input_layer/C13_embedding/C13_embedding_weights/TileYinput_layer/sparse_input_layer/input_layer/C13_embedding/C13_embedding_weights/zeros_likefinput_layer/sparse_input_layer/input_layer/C13_embedding/C13_embedding_weights/embedding_lookup_sparse*'
_output_shapes
:€€€€€€€€€*
T0
з
Uinput_layer/sparse_input_layer/input_layer/C13_embedding/C13_embedding_weights/Cast_1CastTinput_layer/sparse_input_layer/input_layer/C13_embedding/to_sparse_input/dense_shape*
_output_shapes
:*

SrcT0	*

DstT0
¶
\input_layer/sparse_input_layer/input_layer/C13_embedding/C13_embedding_weights/Slice_1/beginConst*
_output_shapes
:*
dtype0*
valueB: 
•
[input_layer/sparse_input_layer/input_layer/C13_embedding/C13_embedding_weights/Slice_1/sizeConst*
dtype0*
_output_shapes
:*
valueB:
£
Vinput_layer/sparse_input_layer/input_layer/C13_embedding/C13_embedding_weights/Slice_1SliceUinput_layer/sparse_input_layer/input_layer/C13_embedding/C13_embedding_weights/Cast_1\input_layer/sparse_input_layer/input_layer/C13_embedding/C13_embedding_weights/Slice_1/begin[input_layer/sparse_input_layer/input_layer/C13_embedding/C13_embedding_weights/Slice_1/size*
Index0*
T0*
_output_shapes
:
‘
Vinput_layer/sparse_input_layer/input_layer/C13_embedding/C13_embedding_weights/Shape_1ShapeNinput_layer/sparse_input_layer/input_layer/C13_embedding/C13_embedding_weights*
T0*
_output_shapes
:
¶
\input_layer/sparse_input_layer/input_layer/C13_embedding/C13_embedding_weights/Slice_2/beginConst*
_output_shapes
:*
valueB:*
dtype0
Ѓ
[input_layer/sparse_input_layer/input_layer/C13_embedding/C13_embedding_weights/Slice_2/sizeConst*
valueB:
€€€€€€€€€*
dtype0*
_output_shapes
:
§
Vinput_layer/sparse_input_layer/input_layer/C13_embedding/C13_embedding_weights/Slice_2SliceVinput_layer/sparse_input_layer/input_layer/C13_embedding/C13_embedding_weights/Shape_1\input_layer/sparse_input_layer/input_layer/C13_embedding/C13_embedding_weights/Slice_2/begin[input_layer/sparse_input_layer/input_layer/C13_embedding/C13_embedding_weights/Slice_2/size*
T0*
_output_shapes
:*
Index0
Ь
Zinput_layer/sparse_input_layer/input_layer/C13_embedding/C13_embedding_weights/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
Ы
Uinput_layer/sparse_input_layer/input_layer/C13_embedding/C13_embedding_weights/concatConcatV2Vinput_layer/sparse_input_layer/input_layer/C13_embedding/C13_embedding_weights/Slice_1Vinput_layer/sparse_input_layer/input_layer/C13_embedding/C13_embedding_weights/Slice_2Zinput_layer/sparse_input_layer/input_layer/C13_embedding/C13_embedding_weights/concat/axis*
_output_shapes
:*
T0*
N
Љ
Xinput_layer/sparse_input_layer/input_layer/C13_embedding/C13_embedding_weights/Reshape_2ReshapeNinput_layer/sparse_input_layer/input_layer/C13_embedding/C13_embedding_weightsUinput_layer/sparse_input_layer/input_layer/C13_embedding/C13_embedding_weights/concat*'
_output_shapes
:€€€€€€€€€*
T0
∆
>input_layer/sparse_input_layer/input_layer/C13_embedding/ShapeShapeXinput_layer/sparse_input_layer/input_layer/C13_embedding/C13_embedding_weights/Reshape_2*
_output_shapes
:*
T0
Ц
Linput_layer/sparse_input_layer/input_layer/C13_embedding/strided_slice/stackConst*
dtype0*
valueB: *
_output_shapes
:
Ш
Ninput_layer/sparse_input_layer/input_layer/C13_embedding/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
Ш
Ninput_layer/sparse_input_layer/input_layer/C13_embedding/strided_slice/stack_2Const*
_output_shapes
:*
valueB:*
dtype0
 
Finput_layer/sparse_input_layer/input_layer/C13_embedding/strided_sliceStridedSlice>input_layer/sparse_input_layer/input_layer/C13_embedding/ShapeLinput_layer/sparse_input_layer/input_layer/C13_embedding/strided_slice/stackNinput_layer/sparse_input_layer/input_layer/C13_embedding/strided_slice/stack_1Ninput_layer/sparse_input_layer/input_layer/C13_embedding/strided_slice/stack_2*
Index0*
T0*
shrink_axis_mask*
_output_shapes
: 
К
Hinput_layer/sparse_input_layer/input_layer/C13_embedding/Reshape/shape/1Const*
value	B :*
_output_shapes
: *
dtype0
О
Finput_layer/sparse_input_layer/input_layer/C13_embedding/Reshape/shapePackFinput_layer/sparse_input_layer/input_layer/C13_embedding/strided_sliceHinput_layer/sparse_input_layer/input_layer/C13_embedding/Reshape/shape/1*
T0*
N*
_output_shapes
:
Я
@input_layer/sparse_input_layer/input_layer/C13_embedding/ReshapeReshapeXinput_layer/sparse_input_layer/input_layer/C13_embedding/C13_embedding_weights/Reshape_2Finput_layer/sparse_input_layer/input_layer/C13_embedding/Reshape/shape*'
_output_shapes
:€€€€€€€€€*
T0
Т
Ginput_layer/sparse_input_layer/input_layer/C14_embedding/ExpandDims/dimConst*
valueB :
€€€€€€€€€*
dtype0*
_output_shapes
: 
№
Cinput_layer/sparse_input_layer/input_layer/C14_embedding/ExpandDims
ExpandDimsPlaceholder_26Ginput_layer/sparse_input_layer/input_layer/C14_embedding/ExpandDims/dim*'
_output_shapes
:€€€€€€€€€*
T0
Ш
Winput_layer/sparse_input_layer/input_layer/C14_embedding/to_sparse_input/ignore_value/xConst*
valueB B *
dtype0*
_output_shapes
: 
≠
Qinput_layer/sparse_input_layer/input_layer/C14_embedding/to_sparse_input/NotEqualNotEqualCinput_layer/sparse_input_layer/input_layer/C14_embedding/ExpandDimsWinput_layer/sparse_input_layer/input_layer/C14_embedding/to_sparse_input/ignore_value/x*'
_output_shapes
:€€€€€€€€€*
T0
’
Pinput_layer/sparse_input_layer/input_layer/C14_embedding/to_sparse_input/indicesWhereQinput_layer/sparse_input_layer/input_layer/C14_embedding/to_sparse_input/NotEqual*'
_output_shapes
:€€€€€€€€€
ґ
Oinput_layer/sparse_input_layer/input_layer/C14_embedding/to_sparse_input/valuesGatherNdCinput_layer/sparse_input_layer/input_layer/C14_embedding/ExpandDimsPinput_layer/sparse_input_layer/input_layer/C14_embedding/to_sparse_input/indices*#
_output_shapes
:€€€€€€€€€*
Tindices0	*
Tparams0
„
Tinput_layer/sparse_input_layer/input_layer/C14_embedding/to_sparse_input/dense_shapeShapeCinput_layer/sparse_input_layer/input_layer/C14_embedding/ExpandDims*
_output_shapes
:*
T0*
out_type0	
г
?input_layer/sparse_input_layer/input_layer/C14_embedding/lookupStringToHashBucketFastOinput_layer/sparse_input_layer/input_layer/C14_embedding/to_sparse_input/values*#
_output_shapes
:€€€€€€€€€*
num_bucketsРN
Э
minput_layer/sparse_input_layer/input_layer/C14_embedding/embedding_weights/Initializer/truncated_normal/shapeConst*
_output_shapes
:*
dtype0*
valueB"'     *]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C14_embedding/embedding_weights
Р
linput_layer/sparse_input_layer/input_layer/C14_embedding/embedding_weights/Initializer/truncated_normal/meanConst*
_output_shapes
: *
dtype0*]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C14_embedding/embedding_weights*
valueB
 *    
Т
ninput_layer/sparse_input_layer/input_layer/C14_embedding/embedding_weights/Initializer/truncated_normal/stddevConst*
valueB
 *  А>*
dtype0*]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C14_embedding/embedding_weights*
_output_shapes
: 
П
winput_layer/sparse_input_layer/input_layer/C14_embedding/embedding_weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormalminput_layer/sparse_input_layer/input_layer/C14_embedding/embedding_weights/Initializer/truncated_normal/shape*
dtype0*]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C14_embedding/embedding_weights*
_output_shapes
:	РN*
T0
д
kinput_layer/sparse_input_layer/input_layer/C14_embedding/embedding_weights/Initializer/truncated_normal/mulMulwinput_layer/sparse_input_layer/input_layer/C14_embedding/embedding_weights/Initializer/truncated_normal/TruncatedNormalninput_layer/sparse_input_layer/input_layer/C14_embedding/embedding_weights/Initializer/truncated_normal/stddev*]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C14_embedding/embedding_weights*
T0*
_output_shapes
:	РN
“
ginput_layer/sparse_input_layer/input_layer/C14_embedding/embedding_weights/Initializer/truncated_normalAddkinput_layer/sparse_input_layer/input_layer/C14_embedding/embedding_weights/Initializer/truncated_normal/mullinput_layer/sparse_input_layer/input_layer/C14_embedding/embedding_weights/Initializer/truncated_normal/mean*
T0*]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C14_embedding/embedding_weights*
_output_shapes
:	РN
ы
Jinput_layer/sparse_input_layer/input_layer/C14_embedding/embedding_weights
VariableV2*
dtype0*]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C14_embedding/embedding_weights*
shape:	РN*
_output_shapes
:	РN
Щ
Qinput_layer/sparse_input_layer/input_layer/C14_embedding/embedding_weights/AssignAssignJinput_layer/sparse_input_layer/input_layer/C14_embedding/embedding_weightsginput_layer/sparse_input_layer/input_layer/C14_embedding/embedding_weights/Initializer/truncated_normal*]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C14_embedding/embedding_weights*
T0*
_output_shapes
:	РN
∞
Oinput_layer/sparse_input_layer/input_layer/C14_embedding/embedding_weights/readIdentityJinput_layer/sparse_input_layer/input_layer/C14_embedding/embedding_weights*
T0*
_output_shapes
:	РN*]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C14_embedding/embedding_weights
§
Zinput_layer/sparse_input_layer/input_layer/C14_embedding/C14_embedding_weights/Slice/beginConst*
valueB: *
_output_shapes
:*
dtype0
£
Yinput_layer/sparse_input_layer/input_layer/C14_embedding/C14_embedding_weights/Slice/sizeConst*
valueB:*
dtype0*
_output_shapes
:
Ь
Tinput_layer/sparse_input_layer/input_layer/C14_embedding/C14_embedding_weights/SliceSliceTinput_layer/sparse_input_layer/input_layer/C14_embedding/to_sparse_input/dense_shapeZinput_layer/sparse_input_layer/input_layer/C14_embedding/C14_embedding_weights/Slice/beginYinput_layer/sparse_input_layer/input_layer/C14_embedding/C14_embedding_weights/Slice/size*
T0	*
_output_shapes
:*
Index0
Ю
Tinput_layer/sparse_input_layer/input_layer/C14_embedding/C14_embedding_weights/ConstConst*
dtype0*
_output_shapes
:*
valueB: 
®
Sinput_layer/sparse_input_layer/input_layer/C14_embedding/C14_embedding_weights/ProdProdTinput_layer/sparse_input_layer/input_layer/C14_embedding/C14_embedding_weights/SliceTinput_layer/sparse_input_layer/input_layer/C14_embedding/C14_embedding_weights/Const*
T0	*
_output_shapes
: 
°
_input_layer/sparse_input_layer/input_layer/C14_embedding/C14_embedding_weights/GatherV2/indicesConst*
value	B :*
_output_shapes
: *
dtype0
Ю
\input_layer/sparse_input_layer/input_layer/C14_embedding/C14_embedding_weights/GatherV2/axisConst*
dtype0*
_output_shapes
: *
value	B : 
Љ
Winput_layer/sparse_input_layer/input_layer/C14_embedding/C14_embedding_weights/GatherV2GatherV2Tinput_layer/sparse_input_layer/input_layer/C14_embedding/to_sparse_input/dense_shape_input_layer/sparse_input_layer/input_layer/C14_embedding/C14_embedding_weights/GatherV2/indices\input_layer/sparse_input_layer/input_layer/C14_embedding/C14_embedding_weights/GatherV2/axis*
Tindices0*
Tparams0	*
_output_shapes
: *
Taxis0
є
Uinput_layer/sparse_input_layer/input_layer/C14_embedding/C14_embedding_weights/Cast/xPackSinput_layer/sparse_input_layer/input_layer/C14_embedding/C14_embedding_weights/ProdWinput_layer/sparse_input_layer/input_layer/C14_embedding/C14_embedding_weights/GatherV2*
T0	*
N*
_output_shapes
:
Ы
\input_layer/sparse_input_layer/input_layer/C14_embedding/C14_embedding_weights/SparseReshapeSparseReshapePinput_layer/sparse_input_layer/input_layer/C14_embedding/to_sparse_input/indicesTinput_layer/sparse_input_layer/input_layer/C14_embedding/to_sparse_input/dense_shapeUinput_layer/sparse_input_layer/input_layer/C14_embedding/C14_embedding_weights/Cast/x*-
_output_shapes
:€€€€€€€€€:
а
einput_layer/sparse_input_layer/input_layer/C14_embedding/C14_embedding_weights/SparseReshape/IdentityIdentity?input_layer/sparse_input_layer/input_layer/C14_embedding/lookup*
T0	*#
_output_shapes
:€€€€€€€€€
Я
]input_layer/sparse_input_layer/input_layer/C14_embedding/C14_embedding_weights/GreaterEqual/yConst*
dtype0	*
_output_shapes
: *
value	B	 R 
я
[input_layer/sparse_input_layer/input_layer/C14_embedding/C14_embedding_weights/GreaterEqualGreaterEqualeinput_layer/sparse_input_layer/input_layer/C14_embedding/C14_embedding_weights/SparseReshape/Identity]input_layer/sparse_input_layer/input_layer/C14_embedding/C14_embedding_weights/GreaterEqual/y*#
_output_shapes
:€€€€€€€€€*
T0	
г
Tinput_layer/sparse_input_layer/input_layer/C14_embedding/C14_embedding_weights/WhereWhere[input_layer/sparse_input_layer/input_layer/C14_embedding/C14_embedding_weights/GreaterEqual*'
_output_shapes
:€€€€€€€€€
ѓ
\input_layer/sparse_input_layer/input_layer/C14_embedding/C14_embedding_weights/Reshape/shapeConst*
_output_shapes
:*
valueB:
€€€€€€€€€*
dtype0
√
Vinput_layer/sparse_input_layer/input_layer/C14_embedding/C14_embedding_weights/ReshapeReshapeTinput_layer/sparse_input_layer/input_layer/C14_embedding/C14_embedding_weights/Where\input_layer/sparse_input_layer/input_layer/C14_embedding/C14_embedding_weights/Reshape/shape*
T0	*#
_output_shapes
:€€€€€€€€€
†
^input_layer/sparse_input_layer/input_layer/C14_embedding/C14_embedding_weights/GatherV2_1/axisConst*
dtype0*
value	B : *
_output_shapes
: 
–
Yinput_layer/sparse_input_layer/input_layer/C14_embedding/C14_embedding_weights/GatherV2_1GatherV2\input_layer/sparse_input_layer/input_layer/C14_embedding/C14_embedding_weights/SparseReshapeVinput_layer/sparse_input_layer/input_layer/C14_embedding/C14_embedding_weights/Reshape^input_layer/sparse_input_layer/input_layer/C14_embedding/C14_embedding_weights/GatherV2_1/axis*
Taxis0*
Tparams0	*'
_output_shapes
:€€€€€€€€€*
Tindices0	
†
^input_layer/sparse_input_layer/input_layer/C14_embedding/C14_embedding_weights/GatherV2_2/axisConst*
dtype0*
value	B : *
_output_shapes
: 
’
Yinput_layer/sparse_input_layer/input_layer/C14_embedding/C14_embedding_weights/GatherV2_2GatherV2einput_layer/sparse_input_layer/input_layer/C14_embedding/C14_embedding_weights/SparseReshape/IdentityVinput_layer/sparse_input_layer/input_layer/C14_embedding/C14_embedding_weights/Reshape^input_layer/sparse_input_layer/input_layer/C14_embedding/C14_embedding_weights/GatherV2_2/axis*
Tindices0	*
Taxis0*
Tparams0	*#
_output_shapes
:€€€€€€€€€
и
Winput_layer/sparse_input_layer/input_layer/C14_embedding/C14_embedding_weights/IdentityIdentity^input_layer/sparse_input_layer/input_layer/C14_embedding/C14_embedding_weights/SparseReshape:1*
_output_shapes
:*
T0	
™
hinput_layer/sparse_input_layer/input_layer/C14_embedding/C14_embedding_weights/SparseFillEmptyRows/ConstConst*
dtype0	*
value	B	 R *
_output_shapes
: 
е
vinput_layer/sparse_input_layer/input_layer/C14_embedding/C14_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowsSparseFillEmptyRowsYinput_layer/sparse_input_layer/input_layer/C14_embedding/C14_embedding_weights/GatherV2_1Yinput_layer/sparse_input_layer/input_layer/C14_embedding/C14_embedding_weights/GatherV2_2Winput_layer/sparse_input_layer/input_layer/C14_embedding/C14_embedding_weights/Identityhinput_layer/sparse_input_layer/input_layer/C14_embedding/C14_embedding_weights/SparseFillEmptyRows/Const*
T0	*T
_output_shapesB
@:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€
Ћ
zinput_layer/sparse_input_layer/input_layer/C14_embedding/C14_embedding_weights/embedding_lookup_sparse/strided_slice/stackConst*
dtype0*
valueB"        *
_output_shapes
:
Ќ
|input_layer/sparse_input_layer/input_layer/C14_embedding/C14_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1Const*
_output_shapes
:*
valueB"       *
dtype0
Ќ
|input_layer/sparse_input_layer/input_layer/C14_embedding/C14_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2Const*
valueB"      *
_output_shapes
:*
dtype0
й
tinput_layer/sparse_input_layer/input_layer/C14_embedding/C14_embedding_weights/embedding_lookup_sparse/strided_sliceStridedSlicevinput_layer/sparse_input_layer/input_layer/C14_embedding/C14_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowszinput_layer/sparse_input_layer/input_layer/C14_embedding/C14_embedding_weights/embedding_lookup_sparse/strided_slice/stack|input_layer/sparse_input_layer/input_layer/C14_embedding/C14_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1|input_layer/sparse_input_layer/input_layer/C14_embedding/C14_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2*
Index0*
T0	*

begin_mask*
end_mask*#
_output_shapes
:€€€€€€€€€*
shrink_axis_mask
¶
kinput_layer/sparse_input_layer/input_layer/C14_embedding/C14_embedding_weights/embedding_lookup_sparse/CastCasttinput_layer/sparse_input_layer/input_layer/C14_embedding/C14_embedding_weights/embedding_lookup_sparse/strided_slice*#
_output_shapes
:€€€€€€€€€*

SrcT0	*

DstT0
Ѓ
minput_layer/sparse_input_layer/input_layer/C14_embedding/C14_embedding_weights/embedding_lookup_sparse/UniqueUniquexinput_layer/sparse_input_layer/input_layer/C14_embedding/C14_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:1*
T0	*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
Э
|input_layer/sparse_input_layer/input_layer/C14_embedding/C14_embedding_weights/embedding_lookup_sparse/embedding_lookup/axisConst*
value	B : *
_output_shapes
: *]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C14_embedding/embedding_weights*
dtype0
х
winput_layer/sparse_input_layer/input_layer/C14_embedding/C14_embedding_weights/embedding_lookup_sparse/embedding_lookupGatherV2Oinput_layer/sparse_input_layer/input_layer/C14_embedding/embedding_weights/readminput_layer/sparse_input_layer/input_layer/C14_embedding/C14_embedding_weights/embedding_lookup_sparse/Unique|input_layer/sparse_input_layer/input_layer/C14_embedding/C14_embedding_weights/embedding_lookup_sparse/embedding_lookup/axis*'
_output_shapes
:€€€€€€€€€*
Taxis0*]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C14_embedding/embedding_weights*
Tindices0	*
Tparams0
Є
Аinput_layer/sparse_input_layer/input_layer/C14_embedding/C14_embedding_weights/embedding_lookup_sparse/embedding_lookup/IdentityIdentitywinput_layer/sparse_input_layer/input_layer/C14_embedding/C14_embedding_weights/embedding_lookup_sparse/embedding_lookup*
T0*'
_output_shapes
:€€€€€€€€€
О
finput_layer/sparse_input_layer/input_layer/C14_embedding/C14_embedding_weights/embedding_lookup_sparseSparseSegmentMeanАinput_layer/sparse_input_layer/input_layer/C14_embedding/C14_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identityoinput_layer/sparse_input_layer/input_layer/C14_embedding/C14_embedding_weights/embedding_lookup_sparse/Unique:1kinput_layer/sparse_input_layer/input_layer/C14_embedding/C14_embedding_weights/embedding_lookup_sparse/Cast*'
_output_shapes
:€€€€€€€€€*
T0
ѓ
^input_layer/sparse_input_layer/input_layer/C14_embedding/C14_embedding_weights/Reshape_1/shapeConst*
valueB"€€€€   *
dtype0*
_output_shapes
:
п
Xinput_layer/sparse_input_layer/input_layer/C14_embedding/C14_embedding_weights/Reshape_1Reshapexinput_layer/sparse_input_layer/input_layer/C14_embedding/C14_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:2^input_layer/sparse_input_layer/input_layer/C14_embedding/C14_embedding_weights/Reshape_1/shape*
T0
*'
_output_shapes
:€€€€€€€€€
к
Tinput_layer/sparse_input_layer/input_layer/C14_embedding/C14_embedding_weights/ShapeShapefinput_layer/sparse_input_layer/input_layer/C14_embedding/C14_embedding_weights/embedding_lookup_sparse*
T0*
_output_shapes
:
ђ
binput_layer/sparse_input_layer/input_layer/C14_embedding/C14_embedding_weights/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB:
Ѓ
dinput_layer/sparse_input_layer/input_layer/C14_embedding/C14_embedding_weights/strided_slice/stack_1Const*
valueB:*
_output_shapes
:*
dtype0
Ѓ
dinput_layer/sparse_input_layer/input_layer/C14_embedding/C14_embedding_weights/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
Є
\input_layer/sparse_input_layer/input_layer/C14_embedding/C14_embedding_weights/strided_sliceStridedSliceTinput_layer/sparse_input_layer/input_layer/C14_embedding/C14_embedding_weights/Shapebinput_layer/sparse_input_layer/input_layer/C14_embedding/C14_embedding_weights/strided_slice/stackdinput_layer/sparse_input_layer/input_layer/C14_embedding/C14_embedding_weights/strided_slice/stack_1dinput_layer/sparse_input_layer/input_layer/C14_embedding/C14_embedding_weights/strided_slice/stack_2*
shrink_axis_mask*
T0*
Index0*
_output_shapes
: 
Ш
Vinput_layer/sparse_input_layer/input_layer/C14_embedding/C14_embedding_weights/stack/0Const*
_output_shapes
: *
dtype0*
value	B :
ј
Tinput_layer/sparse_input_layer/input_layer/C14_embedding/C14_embedding_weights/stackPackVinput_layer/sparse_input_layer/input_layer/C14_embedding/C14_embedding_weights/stack/0\input_layer/sparse_input_layer/input_layer/C14_embedding/C14_embedding_weights/strided_slice*
_output_shapes
:*
N*
T0
∆
Sinput_layer/sparse_input_layer/input_layer/C14_embedding/C14_embedding_weights/TileTileXinput_layer/sparse_input_layer/input_layer/C14_embedding/C14_embedding_weights/Reshape_1Tinput_layer/sparse_input_layer/input_layer/C14_embedding/C14_embedding_weights/stack*
T0
*0
_output_shapes
:€€€€€€€€€€€€€€€€€€
А
Yinput_layer/sparse_input_layer/input_layer/C14_embedding/C14_embedding_weights/zeros_like	ZerosLikefinput_layer/sparse_input_layer/input_layer/C14_embedding/C14_embedding_weights/embedding_lookup_sparse*'
_output_shapes
:€€€€€€€€€*
T0
Ґ
Ninput_layer/sparse_input_layer/input_layer/C14_embedding/C14_embedding_weightsSelectSinput_layer/sparse_input_layer/input_layer/C14_embedding/C14_embedding_weights/TileYinput_layer/sparse_input_layer/input_layer/C14_embedding/C14_embedding_weights/zeros_likefinput_layer/sparse_input_layer/input_layer/C14_embedding/C14_embedding_weights/embedding_lookup_sparse*'
_output_shapes
:€€€€€€€€€*
T0
з
Uinput_layer/sparse_input_layer/input_layer/C14_embedding/C14_embedding_weights/Cast_1CastTinput_layer/sparse_input_layer/input_layer/C14_embedding/to_sparse_input/dense_shape*
_output_shapes
:*

DstT0*

SrcT0	
¶
\input_layer/sparse_input_layer/input_layer/C14_embedding/C14_embedding_weights/Slice_1/beginConst*
_output_shapes
:*
dtype0*
valueB: 
•
[input_layer/sparse_input_layer/input_layer/C14_embedding/C14_embedding_weights/Slice_1/sizeConst*
_output_shapes
:*
dtype0*
valueB:
£
Vinput_layer/sparse_input_layer/input_layer/C14_embedding/C14_embedding_weights/Slice_1SliceUinput_layer/sparse_input_layer/input_layer/C14_embedding/C14_embedding_weights/Cast_1\input_layer/sparse_input_layer/input_layer/C14_embedding/C14_embedding_weights/Slice_1/begin[input_layer/sparse_input_layer/input_layer/C14_embedding/C14_embedding_weights/Slice_1/size*
_output_shapes
:*
Index0*
T0
‘
Vinput_layer/sparse_input_layer/input_layer/C14_embedding/C14_embedding_weights/Shape_1ShapeNinput_layer/sparse_input_layer/input_layer/C14_embedding/C14_embedding_weights*
T0*
_output_shapes
:
¶
\input_layer/sparse_input_layer/input_layer/C14_embedding/C14_embedding_weights/Slice_2/beginConst*
_output_shapes
:*
valueB:*
dtype0
Ѓ
[input_layer/sparse_input_layer/input_layer/C14_embedding/C14_embedding_weights/Slice_2/sizeConst*
dtype0*
_output_shapes
:*
valueB:
€€€€€€€€€
§
Vinput_layer/sparse_input_layer/input_layer/C14_embedding/C14_embedding_weights/Slice_2SliceVinput_layer/sparse_input_layer/input_layer/C14_embedding/C14_embedding_weights/Shape_1\input_layer/sparse_input_layer/input_layer/C14_embedding/C14_embedding_weights/Slice_2/begin[input_layer/sparse_input_layer/input_layer/C14_embedding/C14_embedding_weights/Slice_2/size*
_output_shapes
:*
Index0*
T0
Ь
Zinput_layer/sparse_input_layer/input_layer/C14_embedding/C14_embedding_weights/concat/axisConst*
_output_shapes
: *
value	B : *
dtype0
Ы
Uinput_layer/sparse_input_layer/input_layer/C14_embedding/C14_embedding_weights/concatConcatV2Vinput_layer/sparse_input_layer/input_layer/C14_embedding/C14_embedding_weights/Slice_1Vinput_layer/sparse_input_layer/input_layer/C14_embedding/C14_embedding_weights/Slice_2Zinput_layer/sparse_input_layer/input_layer/C14_embedding/C14_embedding_weights/concat/axis*
_output_shapes
:*
N*
T0
Љ
Xinput_layer/sparse_input_layer/input_layer/C14_embedding/C14_embedding_weights/Reshape_2ReshapeNinput_layer/sparse_input_layer/input_layer/C14_embedding/C14_embedding_weightsUinput_layer/sparse_input_layer/input_layer/C14_embedding/C14_embedding_weights/concat*'
_output_shapes
:€€€€€€€€€*
T0
∆
>input_layer/sparse_input_layer/input_layer/C14_embedding/ShapeShapeXinput_layer/sparse_input_layer/input_layer/C14_embedding/C14_embedding_weights/Reshape_2*
_output_shapes
:*
T0
Ц
Linput_layer/sparse_input_layer/input_layer/C14_embedding/strided_slice/stackConst*
valueB: *
_output_shapes
:*
dtype0
Ш
Ninput_layer/sparse_input_layer/input_layer/C14_embedding/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
Ш
Ninput_layer/sparse_input_layer/input_layer/C14_embedding/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
 
Finput_layer/sparse_input_layer/input_layer/C14_embedding/strided_sliceStridedSlice>input_layer/sparse_input_layer/input_layer/C14_embedding/ShapeLinput_layer/sparse_input_layer/input_layer/C14_embedding/strided_slice/stackNinput_layer/sparse_input_layer/input_layer/C14_embedding/strided_slice/stack_1Ninput_layer/sparse_input_layer/input_layer/C14_embedding/strided_slice/stack_2*
_output_shapes
: *
shrink_axis_mask*
Index0*
T0
К
Hinput_layer/sparse_input_layer/input_layer/C14_embedding/Reshape/shape/1Const*
_output_shapes
: *
value	B :*
dtype0
О
Finput_layer/sparse_input_layer/input_layer/C14_embedding/Reshape/shapePackFinput_layer/sparse_input_layer/input_layer/C14_embedding/strided_sliceHinput_layer/sparse_input_layer/input_layer/C14_embedding/Reshape/shape/1*
T0*
N*
_output_shapes
:
Я
@input_layer/sparse_input_layer/input_layer/C14_embedding/ReshapeReshapeXinput_layer/sparse_input_layer/input_layer/C14_embedding/C14_embedding_weights/Reshape_2Finput_layer/sparse_input_layer/input_layer/C14_embedding/Reshape/shape*'
_output_shapes
:€€€€€€€€€*
T0
Т
Ginput_layer/sparse_input_layer/input_layer/C15_embedding/ExpandDims/dimConst*
valueB :
€€€€€€€€€*
_output_shapes
: *
dtype0
№
Cinput_layer/sparse_input_layer/input_layer/C15_embedding/ExpandDims
ExpandDimsPlaceholder_27Ginput_layer/sparse_input_layer/input_layer/C15_embedding/ExpandDims/dim*
T0*'
_output_shapes
:€€€€€€€€€
Ш
Winput_layer/sparse_input_layer/input_layer/C15_embedding/to_sparse_input/ignore_value/xConst*
valueB B *
dtype0*
_output_shapes
: 
≠
Qinput_layer/sparse_input_layer/input_layer/C15_embedding/to_sparse_input/NotEqualNotEqualCinput_layer/sparse_input_layer/input_layer/C15_embedding/ExpandDimsWinput_layer/sparse_input_layer/input_layer/C15_embedding/to_sparse_input/ignore_value/x*
T0*'
_output_shapes
:€€€€€€€€€
’
Pinput_layer/sparse_input_layer/input_layer/C15_embedding/to_sparse_input/indicesWhereQinput_layer/sparse_input_layer/input_layer/C15_embedding/to_sparse_input/NotEqual*'
_output_shapes
:€€€€€€€€€
ґ
Oinput_layer/sparse_input_layer/input_layer/C15_embedding/to_sparse_input/valuesGatherNdCinput_layer/sparse_input_layer/input_layer/C15_embedding/ExpandDimsPinput_layer/sparse_input_layer/input_layer/C15_embedding/to_sparse_input/indices*
Tparams0*
Tindices0	*#
_output_shapes
:€€€€€€€€€
„
Tinput_layer/sparse_input_layer/input_layer/C15_embedding/to_sparse_input/dense_shapeShapeCinput_layer/sparse_input_layer/input_layer/C15_embedding/ExpandDims*
_output_shapes
:*
out_type0	*
T0
г
?input_layer/sparse_input_layer/input_layer/C15_embedding/lookupStringToHashBucketFastOinput_layer/sparse_input_layer/input_layer/C15_embedding/to_sparse_input/values*
num_bucketsРN*#
_output_shapes
:€€€€€€€€€
Э
minput_layer/sparse_input_layer/input_layer/C15_embedding/embedding_weights/Initializer/truncated_normal/shapeConst*
dtype0*
_output_shapes
:*
valueB"'     *]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C15_embedding/embedding_weights
Р
linput_layer/sparse_input_layer/input_layer/C15_embedding/embedding_weights/Initializer/truncated_normal/meanConst*
dtype0*
valueB
 *    *
_output_shapes
: *]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C15_embedding/embedding_weights
Т
ninput_layer/sparse_input_layer/input_layer/C15_embedding/embedding_weights/Initializer/truncated_normal/stddevConst*
_output_shapes
: *
dtype0*]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C15_embedding/embedding_weights*
valueB
 *  А>
П
winput_layer/sparse_input_layer/input_layer/C15_embedding/embedding_weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormalminput_layer/sparse_input_layer/input_layer/C15_embedding/embedding_weights/Initializer/truncated_normal/shape*
_output_shapes
:	РN*
T0*]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C15_embedding/embedding_weights*
dtype0
д
kinput_layer/sparse_input_layer/input_layer/C15_embedding/embedding_weights/Initializer/truncated_normal/mulMulwinput_layer/sparse_input_layer/input_layer/C15_embedding/embedding_weights/Initializer/truncated_normal/TruncatedNormalninput_layer/sparse_input_layer/input_layer/C15_embedding/embedding_weights/Initializer/truncated_normal/stddev*]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C15_embedding/embedding_weights*
_output_shapes
:	РN*
T0
“
ginput_layer/sparse_input_layer/input_layer/C15_embedding/embedding_weights/Initializer/truncated_normalAddkinput_layer/sparse_input_layer/input_layer/C15_embedding/embedding_weights/Initializer/truncated_normal/mullinput_layer/sparse_input_layer/input_layer/C15_embedding/embedding_weights/Initializer/truncated_normal/mean*
_output_shapes
:	РN*]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C15_embedding/embedding_weights*
T0
ы
Jinput_layer/sparse_input_layer/input_layer/C15_embedding/embedding_weights
VariableV2*
dtype0*
_output_shapes
:	РN*
shape:	РN*]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C15_embedding/embedding_weights
Щ
Qinput_layer/sparse_input_layer/input_layer/C15_embedding/embedding_weights/AssignAssignJinput_layer/sparse_input_layer/input_layer/C15_embedding/embedding_weightsginput_layer/sparse_input_layer/input_layer/C15_embedding/embedding_weights/Initializer/truncated_normal*
T0*
_output_shapes
:	РN*]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C15_embedding/embedding_weights
∞
Oinput_layer/sparse_input_layer/input_layer/C15_embedding/embedding_weights/readIdentityJinput_layer/sparse_input_layer/input_layer/C15_embedding/embedding_weights*
_output_shapes
:	РN*
T0*]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C15_embedding/embedding_weights
§
Zinput_layer/sparse_input_layer/input_layer/C15_embedding/C15_embedding_weights/Slice/beginConst*
_output_shapes
:*
dtype0*
valueB: 
£
Yinput_layer/sparse_input_layer/input_layer/C15_embedding/C15_embedding_weights/Slice/sizeConst*
dtype0*
valueB:*
_output_shapes
:
Ь
Tinput_layer/sparse_input_layer/input_layer/C15_embedding/C15_embedding_weights/SliceSliceTinput_layer/sparse_input_layer/input_layer/C15_embedding/to_sparse_input/dense_shapeZinput_layer/sparse_input_layer/input_layer/C15_embedding/C15_embedding_weights/Slice/beginYinput_layer/sparse_input_layer/input_layer/C15_embedding/C15_embedding_weights/Slice/size*
T0	*
Index0*
_output_shapes
:
Ю
Tinput_layer/sparse_input_layer/input_layer/C15_embedding/C15_embedding_weights/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
®
Sinput_layer/sparse_input_layer/input_layer/C15_embedding/C15_embedding_weights/ProdProdTinput_layer/sparse_input_layer/input_layer/C15_embedding/C15_embedding_weights/SliceTinput_layer/sparse_input_layer/input_layer/C15_embedding/C15_embedding_weights/Const*
_output_shapes
: *
T0	
°
_input_layer/sparse_input_layer/input_layer/C15_embedding/C15_embedding_weights/GatherV2/indicesConst*
dtype0*
_output_shapes
: *
value	B :
Ю
\input_layer/sparse_input_layer/input_layer/C15_embedding/C15_embedding_weights/GatherV2/axisConst*
dtype0*
_output_shapes
: *
value	B : 
Љ
Winput_layer/sparse_input_layer/input_layer/C15_embedding/C15_embedding_weights/GatherV2GatherV2Tinput_layer/sparse_input_layer/input_layer/C15_embedding/to_sparse_input/dense_shape_input_layer/sparse_input_layer/input_layer/C15_embedding/C15_embedding_weights/GatherV2/indices\input_layer/sparse_input_layer/input_layer/C15_embedding/C15_embedding_weights/GatherV2/axis*
Tparams0	*
Taxis0*
Tindices0*
_output_shapes
: 
є
Uinput_layer/sparse_input_layer/input_layer/C15_embedding/C15_embedding_weights/Cast/xPackSinput_layer/sparse_input_layer/input_layer/C15_embedding/C15_embedding_weights/ProdWinput_layer/sparse_input_layer/input_layer/C15_embedding/C15_embedding_weights/GatherV2*
_output_shapes
:*
T0	*
N
Ы
\input_layer/sparse_input_layer/input_layer/C15_embedding/C15_embedding_weights/SparseReshapeSparseReshapePinput_layer/sparse_input_layer/input_layer/C15_embedding/to_sparse_input/indicesTinput_layer/sparse_input_layer/input_layer/C15_embedding/to_sparse_input/dense_shapeUinput_layer/sparse_input_layer/input_layer/C15_embedding/C15_embedding_weights/Cast/x*-
_output_shapes
:€€€€€€€€€:
а
einput_layer/sparse_input_layer/input_layer/C15_embedding/C15_embedding_weights/SparseReshape/IdentityIdentity?input_layer/sparse_input_layer/input_layer/C15_embedding/lookup*#
_output_shapes
:€€€€€€€€€*
T0	
Я
]input_layer/sparse_input_layer/input_layer/C15_embedding/C15_embedding_weights/GreaterEqual/yConst*
_output_shapes
: *
dtype0	*
value	B	 R 
я
[input_layer/sparse_input_layer/input_layer/C15_embedding/C15_embedding_weights/GreaterEqualGreaterEqualeinput_layer/sparse_input_layer/input_layer/C15_embedding/C15_embedding_weights/SparseReshape/Identity]input_layer/sparse_input_layer/input_layer/C15_embedding/C15_embedding_weights/GreaterEqual/y*
T0	*#
_output_shapes
:€€€€€€€€€
г
Tinput_layer/sparse_input_layer/input_layer/C15_embedding/C15_embedding_weights/WhereWhere[input_layer/sparse_input_layer/input_layer/C15_embedding/C15_embedding_weights/GreaterEqual*'
_output_shapes
:€€€€€€€€€
ѓ
\input_layer/sparse_input_layer/input_layer/C15_embedding/C15_embedding_weights/Reshape/shapeConst*
dtype0*
_output_shapes
:*
valueB:
€€€€€€€€€
√
Vinput_layer/sparse_input_layer/input_layer/C15_embedding/C15_embedding_weights/ReshapeReshapeTinput_layer/sparse_input_layer/input_layer/C15_embedding/C15_embedding_weights/Where\input_layer/sparse_input_layer/input_layer/C15_embedding/C15_embedding_weights/Reshape/shape*#
_output_shapes
:€€€€€€€€€*
T0	
†
^input_layer/sparse_input_layer/input_layer/C15_embedding/C15_embedding_weights/GatherV2_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
–
Yinput_layer/sparse_input_layer/input_layer/C15_embedding/C15_embedding_weights/GatherV2_1GatherV2\input_layer/sparse_input_layer/input_layer/C15_embedding/C15_embedding_weights/SparseReshapeVinput_layer/sparse_input_layer/input_layer/C15_embedding/C15_embedding_weights/Reshape^input_layer/sparse_input_layer/input_layer/C15_embedding/C15_embedding_weights/GatherV2_1/axis*'
_output_shapes
:€€€€€€€€€*
Tparams0	*
Taxis0*
Tindices0	
†
^input_layer/sparse_input_layer/input_layer/C15_embedding/C15_embedding_weights/GatherV2_2/axisConst*
dtype0*
_output_shapes
: *
value	B : 
’
Yinput_layer/sparse_input_layer/input_layer/C15_embedding/C15_embedding_weights/GatherV2_2GatherV2einput_layer/sparse_input_layer/input_layer/C15_embedding/C15_embedding_weights/SparseReshape/IdentityVinput_layer/sparse_input_layer/input_layer/C15_embedding/C15_embedding_weights/Reshape^input_layer/sparse_input_layer/input_layer/C15_embedding/C15_embedding_weights/GatherV2_2/axis*#
_output_shapes
:€€€€€€€€€*
Taxis0*
Tindices0	*
Tparams0	
и
Winput_layer/sparse_input_layer/input_layer/C15_embedding/C15_embedding_weights/IdentityIdentity^input_layer/sparse_input_layer/input_layer/C15_embedding/C15_embedding_weights/SparseReshape:1*
_output_shapes
:*
T0	
™
hinput_layer/sparse_input_layer/input_layer/C15_embedding/C15_embedding_weights/SparseFillEmptyRows/ConstConst*
dtype0	*
value	B	 R *
_output_shapes
: 
е
vinput_layer/sparse_input_layer/input_layer/C15_embedding/C15_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowsSparseFillEmptyRowsYinput_layer/sparse_input_layer/input_layer/C15_embedding/C15_embedding_weights/GatherV2_1Yinput_layer/sparse_input_layer/input_layer/C15_embedding/C15_embedding_weights/GatherV2_2Winput_layer/sparse_input_layer/input_layer/C15_embedding/C15_embedding_weights/Identityhinput_layer/sparse_input_layer/input_layer/C15_embedding/C15_embedding_weights/SparseFillEmptyRows/Const*
T0	*T
_output_shapesB
@:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€
Ћ
zinput_layer/sparse_input_layer/input_layer/C15_embedding/C15_embedding_weights/embedding_lookup_sparse/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB"        
Ќ
|input_layer/sparse_input_layer/input_layer/C15_embedding/C15_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1Const*
valueB"       *
_output_shapes
:*
dtype0
Ќ
|input_layer/sparse_input_layer/input_layer/C15_embedding/C15_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:
й
tinput_layer/sparse_input_layer/input_layer/C15_embedding/C15_embedding_weights/embedding_lookup_sparse/strided_sliceStridedSlicevinput_layer/sparse_input_layer/input_layer/C15_embedding/C15_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowszinput_layer/sparse_input_layer/input_layer/C15_embedding/C15_embedding_weights/embedding_lookup_sparse/strided_slice/stack|input_layer/sparse_input_layer/input_layer/C15_embedding/C15_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1|input_layer/sparse_input_layer/input_layer/C15_embedding/C15_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2*
shrink_axis_mask*

begin_mask*
Index0*#
_output_shapes
:€€€€€€€€€*
T0	*
end_mask
¶
kinput_layer/sparse_input_layer/input_layer/C15_embedding/C15_embedding_weights/embedding_lookup_sparse/CastCasttinput_layer/sparse_input_layer/input_layer/C15_embedding/C15_embedding_weights/embedding_lookup_sparse/strided_slice*

DstT0*

SrcT0	*#
_output_shapes
:€€€€€€€€€
Ѓ
minput_layer/sparse_input_layer/input_layer/C15_embedding/C15_embedding_weights/embedding_lookup_sparse/UniqueUniquexinput_layer/sparse_input_layer/input_layer/C15_embedding/C15_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:1*
T0	*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
Э
|input_layer/sparse_input_layer/input_layer/C15_embedding/C15_embedding_weights/embedding_lookup_sparse/embedding_lookup/axisConst*
_output_shapes
: *]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C15_embedding/embedding_weights*
dtype0*
value	B : 
х
winput_layer/sparse_input_layer/input_layer/C15_embedding/C15_embedding_weights/embedding_lookup_sparse/embedding_lookupGatherV2Oinput_layer/sparse_input_layer/input_layer/C15_embedding/embedding_weights/readminput_layer/sparse_input_layer/input_layer/C15_embedding/C15_embedding_weights/embedding_lookup_sparse/Unique|input_layer/sparse_input_layer/input_layer/C15_embedding/C15_embedding_weights/embedding_lookup_sparse/embedding_lookup/axis*
Taxis0*'
_output_shapes
:€€€€€€€€€*
Tparams0*]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C15_embedding/embedding_weights*
Tindices0	
Є
Аinput_layer/sparse_input_layer/input_layer/C15_embedding/C15_embedding_weights/embedding_lookup_sparse/embedding_lookup/IdentityIdentitywinput_layer/sparse_input_layer/input_layer/C15_embedding/C15_embedding_weights/embedding_lookup_sparse/embedding_lookup*'
_output_shapes
:€€€€€€€€€*
T0
О
finput_layer/sparse_input_layer/input_layer/C15_embedding/C15_embedding_weights/embedding_lookup_sparseSparseSegmentMeanАinput_layer/sparse_input_layer/input_layer/C15_embedding/C15_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identityoinput_layer/sparse_input_layer/input_layer/C15_embedding/C15_embedding_weights/embedding_lookup_sparse/Unique:1kinput_layer/sparse_input_layer/input_layer/C15_embedding/C15_embedding_weights/embedding_lookup_sparse/Cast*'
_output_shapes
:€€€€€€€€€*
T0
ѓ
^input_layer/sparse_input_layer/input_layer/C15_embedding/C15_embedding_weights/Reshape_1/shapeConst*
valueB"€€€€   *
_output_shapes
:*
dtype0
п
Xinput_layer/sparse_input_layer/input_layer/C15_embedding/C15_embedding_weights/Reshape_1Reshapexinput_layer/sparse_input_layer/input_layer/C15_embedding/C15_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:2^input_layer/sparse_input_layer/input_layer/C15_embedding/C15_embedding_weights/Reshape_1/shape*'
_output_shapes
:€€€€€€€€€*
T0

к
Tinput_layer/sparse_input_layer/input_layer/C15_embedding/C15_embedding_weights/ShapeShapefinput_layer/sparse_input_layer/input_layer/C15_embedding/C15_embedding_weights/embedding_lookup_sparse*
_output_shapes
:*
T0
ђ
binput_layer/sparse_input_layer/input_layer/C15_embedding/C15_embedding_weights/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
Ѓ
dinput_layer/sparse_input_layer/input_layer/C15_embedding/C15_embedding_weights/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
Ѓ
dinput_layer/sparse_input_layer/input_layer/C15_embedding/C15_embedding_weights/strided_slice/stack_2Const*
valueB:*
_output_shapes
:*
dtype0
Є
\input_layer/sparse_input_layer/input_layer/C15_embedding/C15_embedding_weights/strided_sliceStridedSliceTinput_layer/sparse_input_layer/input_layer/C15_embedding/C15_embedding_weights/Shapebinput_layer/sparse_input_layer/input_layer/C15_embedding/C15_embedding_weights/strided_slice/stackdinput_layer/sparse_input_layer/input_layer/C15_embedding/C15_embedding_weights/strided_slice/stack_1dinput_layer/sparse_input_layer/input_layer/C15_embedding/C15_embedding_weights/strided_slice/stack_2*
_output_shapes
: *
Index0*
shrink_axis_mask*
T0
Ш
Vinput_layer/sparse_input_layer/input_layer/C15_embedding/C15_embedding_weights/stack/0Const*
_output_shapes
: *
dtype0*
value	B :
ј
Tinput_layer/sparse_input_layer/input_layer/C15_embedding/C15_embedding_weights/stackPackVinput_layer/sparse_input_layer/input_layer/C15_embedding/C15_embedding_weights/stack/0\input_layer/sparse_input_layer/input_layer/C15_embedding/C15_embedding_weights/strided_slice*
_output_shapes
:*
N*
T0
∆
Sinput_layer/sparse_input_layer/input_layer/C15_embedding/C15_embedding_weights/TileTileXinput_layer/sparse_input_layer/input_layer/C15_embedding/C15_embedding_weights/Reshape_1Tinput_layer/sparse_input_layer/input_layer/C15_embedding/C15_embedding_weights/stack*0
_output_shapes
:€€€€€€€€€€€€€€€€€€*
T0

А
Yinput_layer/sparse_input_layer/input_layer/C15_embedding/C15_embedding_weights/zeros_like	ZerosLikefinput_layer/sparse_input_layer/input_layer/C15_embedding/C15_embedding_weights/embedding_lookup_sparse*
T0*'
_output_shapes
:€€€€€€€€€
Ґ
Ninput_layer/sparse_input_layer/input_layer/C15_embedding/C15_embedding_weightsSelectSinput_layer/sparse_input_layer/input_layer/C15_embedding/C15_embedding_weights/TileYinput_layer/sparse_input_layer/input_layer/C15_embedding/C15_embedding_weights/zeros_likefinput_layer/sparse_input_layer/input_layer/C15_embedding/C15_embedding_weights/embedding_lookup_sparse*'
_output_shapes
:€€€€€€€€€*
T0
з
Uinput_layer/sparse_input_layer/input_layer/C15_embedding/C15_embedding_weights/Cast_1CastTinput_layer/sparse_input_layer/input_layer/C15_embedding/to_sparse_input/dense_shape*

DstT0*

SrcT0	*
_output_shapes
:
¶
\input_layer/sparse_input_layer/input_layer/C15_embedding/C15_embedding_weights/Slice_1/beginConst*
dtype0*
valueB: *
_output_shapes
:
•
[input_layer/sparse_input_layer/input_layer/C15_embedding/C15_embedding_weights/Slice_1/sizeConst*
_output_shapes
:*
dtype0*
valueB:
£
Vinput_layer/sparse_input_layer/input_layer/C15_embedding/C15_embedding_weights/Slice_1SliceUinput_layer/sparse_input_layer/input_layer/C15_embedding/C15_embedding_weights/Cast_1\input_layer/sparse_input_layer/input_layer/C15_embedding/C15_embedding_weights/Slice_1/begin[input_layer/sparse_input_layer/input_layer/C15_embedding/C15_embedding_weights/Slice_1/size*
T0*
Index0*
_output_shapes
:
‘
Vinput_layer/sparse_input_layer/input_layer/C15_embedding/C15_embedding_weights/Shape_1ShapeNinput_layer/sparse_input_layer/input_layer/C15_embedding/C15_embedding_weights*
T0*
_output_shapes
:
¶
\input_layer/sparse_input_layer/input_layer/C15_embedding/C15_embedding_weights/Slice_2/beginConst*
dtype0*
valueB:*
_output_shapes
:
Ѓ
[input_layer/sparse_input_layer/input_layer/C15_embedding/C15_embedding_weights/Slice_2/sizeConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€
§
Vinput_layer/sparse_input_layer/input_layer/C15_embedding/C15_embedding_weights/Slice_2SliceVinput_layer/sparse_input_layer/input_layer/C15_embedding/C15_embedding_weights/Shape_1\input_layer/sparse_input_layer/input_layer/C15_embedding/C15_embedding_weights/Slice_2/begin[input_layer/sparse_input_layer/input_layer/C15_embedding/C15_embedding_weights/Slice_2/size*
T0*
Index0*
_output_shapes
:
Ь
Zinput_layer/sparse_input_layer/input_layer/C15_embedding/C15_embedding_weights/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
Ы
Uinput_layer/sparse_input_layer/input_layer/C15_embedding/C15_embedding_weights/concatConcatV2Vinput_layer/sparse_input_layer/input_layer/C15_embedding/C15_embedding_weights/Slice_1Vinput_layer/sparse_input_layer/input_layer/C15_embedding/C15_embedding_weights/Slice_2Zinput_layer/sparse_input_layer/input_layer/C15_embedding/C15_embedding_weights/concat/axis*
_output_shapes
:*
T0*
N
Љ
Xinput_layer/sparse_input_layer/input_layer/C15_embedding/C15_embedding_weights/Reshape_2ReshapeNinput_layer/sparse_input_layer/input_layer/C15_embedding/C15_embedding_weightsUinput_layer/sparse_input_layer/input_layer/C15_embedding/C15_embedding_weights/concat*
T0*'
_output_shapes
:€€€€€€€€€
∆
>input_layer/sparse_input_layer/input_layer/C15_embedding/ShapeShapeXinput_layer/sparse_input_layer/input_layer/C15_embedding/C15_embedding_weights/Reshape_2*
_output_shapes
:*
T0
Ц
Linput_layer/sparse_input_layer/input_layer/C15_embedding/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
Ш
Ninput_layer/sparse_input_layer/input_layer/C15_embedding/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
Ш
Ninput_layer/sparse_input_layer/input_layer/C15_embedding/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
 
Finput_layer/sparse_input_layer/input_layer/C15_embedding/strided_sliceStridedSlice>input_layer/sparse_input_layer/input_layer/C15_embedding/ShapeLinput_layer/sparse_input_layer/input_layer/C15_embedding/strided_slice/stackNinput_layer/sparse_input_layer/input_layer/C15_embedding/strided_slice/stack_1Ninput_layer/sparse_input_layer/input_layer/C15_embedding/strided_slice/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
К
Hinput_layer/sparse_input_layer/input_layer/C15_embedding/Reshape/shape/1Const*
value	B :*
_output_shapes
: *
dtype0
О
Finput_layer/sparse_input_layer/input_layer/C15_embedding/Reshape/shapePackFinput_layer/sparse_input_layer/input_layer/C15_embedding/strided_sliceHinput_layer/sparse_input_layer/input_layer/C15_embedding/Reshape/shape/1*
_output_shapes
:*
N*
T0
Я
@input_layer/sparse_input_layer/input_layer/C15_embedding/ReshapeReshapeXinput_layer/sparse_input_layer/input_layer/C15_embedding/C15_embedding_weights/Reshape_2Finput_layer/sparse_input_layer/input_layer/C15_embedding/Reshape/shape*'
_output_shapes
:€€€€€€€€€*
T0
Т
Ginput_layer/sparse_input_layer/input_layer/C16_embedding/ExpandDims/dimConst*
valueB :
€€€€€€€€€*
_output_shapes
: *
dtype0
№
Cinput_layer/sparse_input_layer/input_layer/C16_embedding/ExpandDims
ExpandDimsPlaceholder_28Ginput_layer/sparse_input_layer/input_layer/C16_embedding/ExpandDims/dim*
T0*'
_output_shapes
:€€€€€€€€€
Ш
Winput_layer/sparse_input_layer/input_layer/C16_embedding/to_sparse_input/ignore_value/xConst*
dtype0*
_output_shapes
: *
valueB B 
≠
Qinput_layer/sparse_input_layer/input_layer/C16_embedding/to_sparse_input/NotEqualNotEqualCinput_layer/sparse_input_layer/input_layer/C16_embedding/ExpandDimsWinput_layer/sparse_input_layer/input_layer/C16_embedding/to_sparse_input/ignore_value/x*'
_output_shapes
:€€€€€€€€€*
T0
’
Pinput_layer/sparse_input_layer/input_layer/C16_embedding/to_sparse_input/indicesWhereQinput_layer/sparse_input_layer/input_layer/C16_embedding/to_sparse_input/NotEqual*'
_output_shapes
:€€€€€€€€€
ґ
Oinput_layer/sparse_input_layer/input_layer/C16_embedding/to_sparse_input/valuesGatherNdCinput_layer/sparse_input_layer/input_layer/C16_embedding/ExpandDimsPinput_layer/sparse_input_layer/input_layer/C16_embedding/to_sparse_input/indices*
Tparams0*#
_output_shapes
:€€€€€€€€€*
Tindices0	
„
Tinput_layer/sparse_input_layer/input_layer/C16_embedding/to_sparse_input/dense_shapeShapeCinput_layer/sparse_input_layer/input_layer/C16_embedding/ExpandDims*
T0*
_output_shapes
:*
out_type0	
г
?input_layer/sparse_input_layer/input_layer/C16_embedding/lookupStringToHashBucketFastOinput_layer/sparse_input_layer/input_layer/C16_embedding/to_sparse_input/values*
num_bucketsРN*#
_output_shapes
:€€€€€€€€€
Э
minput_layer/sparse_input_layer/input_layer/C16_embedding/embedding_weights/Initializer/truncated_normal/shapeConst*]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C16_embedding/embedding_weights*
valueB"'     *
_output_shapes
:*
dtype0
Р
linput_layer/sparse_input_layer/input_layer/C16_embedding/embedding_weights/Initializer/truncated_normal/meanConst*
_output_shapes
: *]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C16_embedding/embedding_weights*
dtype0*
valueB
 *    
Т
ninput_layer/sparse_input_layer/input_layer/C16_embedding/embedding_weights/Initializer/truncated_normal/stddevConst*
valueB
 *  А>*
_output_shapes
: *]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C16_embedding/embedding_weights*
dtype0
П
winput_layer/sparse_input_layer/input_layer/C16_embedding/embedding_weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormalminput_layer/sparse_input_layer/input_layer/C16_embedding/embedding_weights/Initializer/truncated_normal/shape*
T0*
_output_shapes
:	РN*]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C16_embedding/embedding_weights*
dtype0
д
kinput_layer/sparse_input_layer/input_layer/C16_embedding/embedding_weights/Initializer/truncated_normal/mulMulwinput_layer/sparse_input_layer/input_layer/C16_embedding/embedding_weights/Initializer/truncated_normal/TruncatedNormalninput_layer/sparse_input_layer/input_layer/C16_embedding/embedding_weights/Initializer/truncated_normal/stddev*
_output_shapes
:	РN*
T0*]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C16_embedding/embedding_weights
“
ginput_layer/sparse_input_layer/input_layer/C16_embedding/embedding_weights/Initializer/truncated_normalAddkinput_layer/sparse_input_layer/input_layer/C16_embedding/embedding_weights/Initializer/truncated_normal/mullinput_layer/sparse_input_layer/input_layer/C16_embedding/embedding_weights/Initializer/truncated_normal/mean*]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C16_embedding/embedding_weights*
T0*
_output_shapes
:	РN
ы
Jinput_layer/sparse_input_layer/input_layer/C16_embedding/embedding_weights
VariableV2*
_output_shapes
:	РN*
dtype0*]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C16_embedding/embedding_weights*
shape:	РN
Щ
Qinput_layer/sparse_input_layer/input_layer/C16_embedding/embedding_weights/AssignAssignJinput_layer/sparse_input_layer/input_layer/C16_embedding/embedding_weightsginput_layer/sparse_input_layer/input_layer/C16_embedding/embedding_weights/Initializer/truncated_normal*]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C16_embedding/embedding_weights*
_output_shapes
:	РN*
T0
∞
Oinput_layer/sparse_input_layer/input_layer/C16_embedding/embedding_weights/readIdentityJinput_layer/sparse_input_layer/input_layer/C16_embedding/embedding_weights*]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C16_embedding/embedding_weights*
T0*
_output_shapes
:	РN
§
Zinput_layer/sparse_input_layer/input_layer/C16_embedding/C16_embedding_weights/Slice/beginConst*
valueB: *
dtype0*
_output_shapes
:
£
Yinput_layer/sparse_input_layer/input_layer/C16_embedding/C16_embedding_weights/Slice/sizeConst*
valueB:*
dtype0*
_output_shapes
:
Ь
Tinput_layer/sparse_input_layer/input_layer/C16_embedding/C16_embedding_weights/SliceSliceTinput_layer/sparse_input_layer/input_layer/C16_embedding/to_sparse_input/dense_shapeZinput_layer/sparse_input_layer/input_layer/C16_embedding/C16_embedding_weights/Slice/beginYinput_layer/sparse_input_layer/input_layer/C16_embedding/C16_embedding_weights/Slice/size*
T0	*
Index0*
_output_shapes
:
Ю
Tinput_layer/sparse_input_layer/input_layer/C16_embedding/C16_embedding_weights/ConstConst*
_output_shapes
:*
valueB: *
dtype0
®
Sinput_layer/sparse_input_layer/input_layer/C16_embedding/C16_embedding_weights/ProdProdTinput_layer/sparse_input_layer/input_layer/C16_embedding/C16_embedding_weights/SliceTinput_layer/sparse_input_layer/input_layer/C16_embedding/C16_embedding_weights/Const*
T0	*
_output_shapes
: 
°
_input_layer/sparse_input_layer/input_layer/C16_embedding/C16_embedding_weights/GatherV2/indicesConst*
dtype0*
value	B :*
_output_shapes
: 
Ю
\input_layer/sparse_input_layer/input_layer/C16_embedding/C16_embedding_weights/GatherV2/axisConst*
dtype0*
value	B : *
_output_shapes
: 
Љ
Winput_layer/sparse_input_layer/input_layer/C16_embedding/C16_embedding_weights/GatherV2GatherV2Tinput_layer/sparse_input_layer/input_layer/C16_embedding/to_sparse_input/dense_shape_input_layer/sparse_input_layer/input_layer/C16_embedding/C16_embedding_weights/GatherV2/indices\input_layer/sparse_input_layer/input_layer/C16_embedding/C16_embedding_weights/GatherV2/axis*
Tparams0	*
Taxis0*
Tindices0*
_output_shapes
: 
є
Uinput_layer/sparse_input_layer/input_layer/C16_embedding/C16_embedding_weights/Cast/xPackSinput_layer/sparse_input_layer/input_layer/C16_embedding/C16_embedding_weights/ProdWinput_layer/sparse_input_layer/input_layer/C16_embedding/C16_embedding_weights/GatherV2*
N*
T0	*
_output_shapes
:
Ы
\input_layer/sparse_input_layer/input_layer/C16_embedding/C16_embedding_weights/SparseReshapeSparseReshapePinput_layer/sparse_input_layer/input_layer/C16_embedding/to_sparse_input/indicesTinput_layer/sparse_input_layer/input_layer/C16_embedding/to_sparse_input/dense_shapeUinput_layer/sparse_input_layer/input_layer/C16_embedding/C16_embedding_weights/Cast/x*-
_output_shapes
:€€€€€€€€€:
а
einput_layer/sparse_input_layer/input_layer/C16_embedding/C16_embedding_weights/SparseReshape/IdentityIdentity?input_layer/sparse_input_layer/input_layer/C16_embedding/lookup*#
_output_shapes
:€€€€€€€€€*
T0	
Я
]input_layer/sparse_input_layer/input_layer/C16_embedding/C16_embedding_weights/GreaterEqual/yConst*
value	B	 R *
_output_shapes
: *
dtype0	
я
[input_layer/sparse_input_layer/input_layer/C16_embedding/C16_embedding_weights/GreaterEqualGreaterEqualeinput_layer/sparse_input_layer/input_layer/C16_embedding/C16_embedding_weights/SparseReshape/Identity]input_layer/sparse_input_layer/input_layer/C16_embedding/C16_embedding_weights/GreaterEqual/y*#
_output_shapes
:€€€€€€€€€*
T0	
г
Tinput_layer/sparse_input_layer/input_layer/C16_embedding/C16_embedding_weights/WhereWhere[input_layer/sparse_input_layer/input_layer/C16_embedding/C16_embedding_weights/GreaterEqual*'
_output_shapes
:€€€€€€€€€
ѓ
\input_layer/sparse_input_layer/input_layer/C16_embedding/C16_embedding_weights/Reshape/shapeConst*
dtype0*
valueB:
€€€€€€€€€*
_output_shapes
:
√
Vinput_layer/sparse_input_layer/input_layer/C16_embedding/C16_embedding_weights/ReshapeReshapeTinput_layer/sparse_input_layer/input_layer/C16_embedding/C16_embedding_weights/Where\input_layer/sparse_input_layer/input_layer/C16_embedding/C16_embedding_weights/Reshape/shape*#
_output_shapes
:€€€€€€€€€*
T0	
†
^input_layer/sparse_input_layer/input_layer/C16_embedding/C16_embedding_weights/GatherV2_1/axisConst*
value	B : *
_output_shapes
: *
dtype0
–
Yinput_layer/sparse_input_layer/input_layer/C16_embedding/C16_embedding_weights/GatherV2_1GatherV2\input_layer/sparse_input_layer/input_layer/C16_embedding/C16_embedding_weights/SparseReshapeVinput_layer/sparse_input_layer/input_layer/C16_embedding/C16_embedding_weights/Reshape^input_layer/sparse_input_layer/input_layer/C16_embedding/C16_embedding_weights/GatherV2_1/axis*
Tindices0	*
Tparams0	*'
_output_shapes
:€€€€€€€€€*
Taxis0
†
^input_layer/sparse_input_layer/input_layer/C16_embedding/C16_embedding_weights/GatherV2_2/axisConst*
dtype0*
value	B : *
_output_shapes
: 
’
Yinput_layer/sparse_input_layer/input_layer/C16_embedding/C16_embedding_weights/GatherV2_2GatherV2einput_layer/sparse_input_layer/input_layer/C16_embedding/C16_embedding_weights/SparseReshape/IdentityVinput_layer/sparse_input_layer/input_layer/C16_embedding/C16_embedding_weights/Reshape^input_layer/sparse_input_layer/input_layer/C16_embedding/C16_embedding_weights/GatherV2_2/axis*
Tindices0	*#
_output_shapes
:€€€€€€€€€*
Taxis0*
Tparams0	
и
Winput_layer/sparse_input_layer/input_layer/C16_embedding/C16_embedding_weights/IdentityIdentity^input_layer/sparse_input_layer/input_layer/C16_embedding/C16_embedding_weights/SparseReshape:1*
_output_shapes
:*
T0	
™
hinput_layer/sparse_input_layer/input_layer/C16_embedding/C16_embedding_weights/SparseFillEmptyRows/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R 
е
vinput_layer/sparse_input_layer/input_layer/C16_embedding/C16_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowsSparseFillEmptyRowsYinput_layer/sparse_input_layer/input_layer/C16_embedding/C16_embedding_weights/GatherV2_1Yinput_layer/sparse_input_layer/input_layer/C16_embedding/C16_embedding_weights/GatherV2_2Winput_layer/sparse_input_layer/input_layer/C16_embedding/C16_embedding_weights/Identityhinput_layer/sparse_input_layer/input_layer/C16_embedding/C16_embedding_weights/SparseFillEmptyRows/Const*T
_output_shapesB
@:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€*
T0	
Ћ
zinput_layer/sparse_input_layer/input_layer/C16_embedding/C16_embedding_weights/embedding_lookup_sparse/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB"        
Ќ
|input_layer/sparse_input_layer/input_layer/C16_embedding/C16_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1Const*
_output_shapes
:*
valueB"       *
dtype0
Ќ
|input_layer/sparse_input_layer/input_layer/C16_embedding/C16_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:
й
tinput_layer/sparse_input_layer/input_layer/C16_embedding/C16_embedding_weights/embedding_lookup_sparse/strided_sliceStridedSlicevinput_layer/sparse_input_layer/input_layer/C16_embedding/C16_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowszinput_layer/sparse_input_layer/input_layer/C16_embedding/C16_embedding_weights/embedding_lookup_sparse/strided_slice/stack|input_layer/sparse_input_layer/input_layer/C16_embedding/C16_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1|input_layer/sparse_input_layer/input_layer/C16_embedding/C16_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2*
shrink_axis_mask*

begin_mask*
T0	*#
_output_shapes
:€€€€€€€€€*
Index0*
end_mask
¶
kinput_layer/sparse_input_layer/input_layer/C16_embedding/C16_embedding_weights/embedding_lookup_sparse/CastCasttinput_layer/sparse_input_layer/input_layer/C16_embedding/C16_embedding_weights/embedding_lookup_sparse/strided_slice*

SrcT0	*

DstT0*#
_output_shapes
:€€€€€€€€€
Ѓ
minput_layer/sparse_input_layer/input_layer/C16_embedding/C16_embedding_weights/embedding_lookup_sparse/UniqueUniquexinput_layer/sparse_input_layer/input_layer/C16_embedding/C16_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€*
T0	
Э
|input_layer/sparse_input_layer/input_layer/C16_embedding/C16_embedding_weights/embedding_lookup_sparse/embedding_lookup/axisConst*
_output_shapes
: *
value	B : *
dtype0*]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C16_embedding/embedding_weights
х
winput_layer/sparse_input_layer/input_layer/C16_embedding/C16_embedding_weights/embedding_lookup_sparse/embedding_lookupGatherV2Oinput_layer/sparse_input_layer/input_layer/C16_embedding/embedding_weights/readminput_layer/sparse_input_layer/input_layer/C16_embedding/C16_embedding_weights/embedding_lookup_sparse/Unique|input_layer/sparse_input_layer/input_layer/C16_embedding/C16_embedding_weights/embedding_lookup_sparse/embedding_lookup/axis*
Taxis0*]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C16_embedding/embedding_weights*
Tindices0	*'
_output_shapes
:€€€€€€€€€*
Tparams0
Є
Аinput_layer/sparse_input_layer/input_layer/C16_embedding/C16_embedding_weights/embedding_lookup_sparse/embedding_lookup/IdentityIdentitywinput_layer/sparse_input_layer/input_layer/C16_embedding/C16_embedding_weights/embedding_lookup_sparse/embedding_lookup*
T0*'
_output_shapes
:€€€€€€€€€
О
finput_layer/sparse_input_layer/input_layer/C16_embedding/C16_embedding_weights/embedding_lookup_sparseSparseSegmentMeanАinput_layer/sparse_input_layer/input_layer/C16_embedding/C16_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identityoinput_layer/sparse_input_layer/input_layer/C16_embedding/C16_embedding_weights/embedding_lookup_sparse/Unique:1kinput_layer/sparse_input_layer/input_layer/C16_embedding/C16_embedding_weights/embedding_lookup_sparse/Cast*'
_output_shapes
:€€€€€€€€€*
T0
ѓ
^input_layer/sparse_input_layer/input_layer/C16_embedding/C16_embedding_weights/Reshape_1/shapeConst*
dtype0*
_output_shapes
:*
valueB"€€€€   
п
Xinput_layer/sparse_input_layer/input_layer/C16_embedding/C16_embedding_weights/Reshape_1Reshapexinput_layer/sparse_input_layer/input_layer/C16_embedding/C16_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:2^input_layer/sparse_input_layer/input_layer/C16_embedding/C16_embedding_weights/Reshape_1/shape*'
_output_shapes
:€€€€€€€€€*
T0

к
Tinput_layer/sparse_input_layer/input_layer/C16_embedding/C16_embedding_weights/ShapeShapefinput_layer/sparse_input_layer/input_layer/C16_embedding/C16_embedding_weights/embedding_lookup_sparse*
_output_shapes
:*
T0
ђ
binput_layer/sparse_input_layer/input_layer/C16_embedding/C16_embedding_weights/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB:
Ѓ
dinput_layer/sparse_input_layer/input_layer/C16_embedding/C16_embedding_weights/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
Ѓ
dinput_layer/sparse_input_layer/input_layer/C16_embedding/C16_embedding_weights/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Є
\input_layer/sparse_input_layer/input_layer/C16_embedding/C16_embedding_weights/strided_sliceStridedSliceTinput_layer/sparse_input_layer/input_layer/C16_embedding/C16_embedding_weights/Shapebinput_layer/sparse_input_layer/input_layer/C16_embedding/C16_embedding_weights/strided_slice/stackdinput_layer/sparse_input_layer/input_layer/C16_embedding/C16_embedding_weights/strided_slice/stack_1dinput_layer/sparse_input_layer/input_layer/C16_embedding/C16_embedding_weights/strided_slice/stack_2*
_output_shapes
: *
Index0*
T0*
shrink_axis_mask
Ш
Vinput_layer/sparse_input_layer/input_layer/C16_embedding/C16_embedding_weights/stack/0Const*
dtype0*
value	B :*
_output_shapes
: 
ј
Tinput_layer/sparse_input_layer/input_layer/C16_embedding/C16_embedding_weights/stackPackVinput_layer/sparse_input_layer/input_layer/C16_embedding/C16_embedding_weights/stack/0\input_layer/sparse_input_layer/input_layer/C16_embedding/C16_embedding_weights/strided_slice*
_output_shapes
:*
N*
T0
∆
Sinput_layer/sparse_input_layer/input_layer/C16_embedding/C16_embedding_weights/TileTileXinput_layer/sparse_input_layer/input_layer/C16_embedding/C16_embedding_weights/Reshape_1Tinput_layer/sparse_input_layer/input_layer/C16_embedding/C16_embedding_weights/stack*0
_output_shapes
:€€€€€€€€€€€€€€€€€€*
T0

А
Yinput_layer/sparse_input_layer/input_layer/C16_embedding/C16_embedding_weights/zeros_like	ZerosLikefinput_layer/sparse_input_layer/input_layer/C16_embedding/C16_embedding_weights/embedding_lookup_sparse*
T0*'
_output_shapes
:€€€€€€€€€
Ґ
Ninput_layer/sparse_input_layer/input_layer/C16_embedding/C16_embedding_weightsSelectSinput_layer/sparse_input_layer/input_layer/C16_embedding/C16_embedding_weights/TileYinput_layer/sparse_input_layer/input_layer/C16_embedding/C16_embedding_weights/zeros_likefinput_layer/sparse_input_layer/input_layer/C16_embedding/C16_embedding_weights/embedding_lookup_sparse*
T0*'
_output_shapes
:€€€€€€€€€
з
Uinput_layer/sparse_input_layer/input_layer/C16_embedding/C16_embedding_weights/Cast_1CastTinput_layer/sparse_input_layer/input_layer/C16_embedding/to_sparse_input/dense_shape*
_output_shapes
:*

DstT0*

SrcT0	
¶
\input_layer/sparse_input_layer/input_layer/C16_embedding/C16_embedding_weights/Slice_1/beginConst*
dtype0*
valueB: *
_output_shapes
:
•
[input_layer/sparse_input_layer/input_layer/C16_embedding/C16_embedding_weights/Slice_1/sizeConst*
dtype0*
_output_shapes
:*
valueB:
£
Vinput_layer/sparse_input_layer/input_layer/C16_embedding/C16_embedding_weights/Slice_1SliceUinput_layer/sparse_input_layer/input_layer/C16_embedding/C16_embedding_weights/Cast_1\input_layer/sparse_input_layer/input_layer/C16_embedding/C16_embedding_weights/Slice_1/begin[input_layer/sparse_input_layer/input_layer/C16_embedding/C16_embedding_weights/Slice_1/size*
_output_shapes
:*
T0*
Index0
‘
Vinput_layer/sparse_input_layer/input_layer/C16_embedding/C16_embedding_weights/Shape_1ShapeNinput_layer/sparse_input_layer/input_layer/C16_embedding/C16_embedding_weights*
_output_shapes
:*
T0
¶
\input_layer/sparse_input_layer/input_layer/C16_embedding/C16_embedding_weights/Slice_2/beginConst*
valueB:*
_output_shapes
:*
dtype0
Ѓ
[input_layer/sparse_input_layer/input_layer/C16_embedding/C16_embedding_weights/Slice_2/sizeConst*
valueB:
€€€€€€€€€*
dtype0*
_output_shapes
:
§
Vinput_layer/sparse_input_layer/input_layer/C16_embedding/C16_embedding_weights/Slice_2SliceVinput_layer/sparse_input_layer/input_layer/C16_embedding/C16_embedding_weights/Shape_1\input_layer/sparse_input_layer/input_layer/C16_embedding/C16_embedding_weights/Slice_2/begin[input_layer/sparse_input_layer/input_layer/C16_embedding/C16_embedding_weights/Slice_2/size*
_output_shapes
:*
Index0*
T0
Ь
Zinput_layer/sparse_input_layer/input_layer/C16_embedding/C16_embedding_weights/concat/axisConst*
dtype0*
value	B : *
_output_shapes
: 
Ы
Uinput_layer/sparse_input_layer/input_layer/C16_embedding/C16_embedding_weights/concatConcatV2Vinput_layer/sparse_input_layer/input_layer/C16_embedding/C16_embedding_weights/Slice_1Vinput_layer/sparse_input_layer/input_layer/C16_embedding/C16_embedding_weights/Slice_2Zinput_layer/sparse_input_layer/input_layer/C16_embedding/C16_embedding_weights/concat/axis*
_output_shapes
:*
T0*
N
Љ
Xinput_layer/sparse_input_layer/input_layer/C16_embedding/C16_embedding_weights/Reshape_2ReshapeNinput_layer/sparse_input_layer/input_layer/C16_embedding/C16_embedding_weightsUinput_layer/sparse_input_layer/input_layer/C16_embedding/C16_embedding_weights/concat*'
_output_shapes
:€€€€€€€€€*
T0
∆
>input_layer/sparse_input_layer/input_layer/C16_embedding/ShapeShapeXinput_layer/sparse_input_layer/input_layer/C16_embedding/C16_embedding_weights/Reshape_2*
T0*
_output_shapes
:
Ц
Linput_layer/sparse_input_layer/input_layer/C16_embedding/strided_slice/stackConst*
dtype0*
valueB: *
_output_shapes
:
Ш
Ninput_layer/sparse_input_layer/input_layer/C16_embedding/strided_slice/stack_1Const*
valueB:*
_output_shapes
:*
dtype0
Ш
Ninput_layer/sparse_input_layer/input_layer/C16_embedding/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
 
Finput_layer/sparse_input_layer/input_layer/C16_embedding/strided_sliceStridedSlice>input_layer/sparse_input_layer/input_layer/C16_embedding/ShapeLinput_layer/sparse_input_layer/input_layer/C16_embedding/strided_slice/stackNinput_layer/sparse_input_layer/input_layer/C16_embedding/strided_slice/stack_1Ninput_layer/sparse_input_layer/input_layer/C16_embedding/strided_slice/stack_2*
Index0*
T0*
shrink_axis_mask*
_output_shapes
: 
К
Hinput_layer/sparse_input_layer/input_layer/C16_embedding/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
О
Finput_layer/sparse_input_layer/input_layer/C16_embedding/Reshape/shapePackFinput_layer/sparse_input_layer/input_layer/C16_embedding/strided_sliceHinput_layer/sparse_input_layer/input_layer/C16_embedding/Reshape/shape/1*
_output_shapes
:*
T0*
N
Я
@input_layer/sparse_input_layer/input_layer/C16_embedding/ReshapeReshapeXinput_layer/sparse_input_layer/input_layer/C16_embedding/C16_embedding_weights/Reshape_2Finput_layer/sparse_input_layer/input_layer/C16_embedding/Reshape/shape*'
_output_shapes
:€€€€€€€€€*
T0
Т
Ginput_layer/sparse_input_layer/input_layer/C17_embedding/ExpandDims/dimConst*
_output_shapes
: *
valueB :
€€€€€€€€€*
dtype0
№
Cinput_layer/sparse_input_layer/input_layer/C17_embedding/ExpandDims
ExpandDimsPlaceholder_29Ginput_layer/sparse_input_layer/input_layer/C17_embedding/ExpandDims/dim*'
_output_shapes
:€€€€€€€€€*
T0
Ш
Winput_layer/sparse_input_layer/input_layer/C17_embedding/to_sparse_input/ignore_value/xConst*
dtype0*
_output_shapes
: *
valueB B 
≠
Qinput_layer/sparse_input_layer/input_layer/C17_embedding/to_sparse_input/NotEqualNotEqualCinput_layer/sparse_input_layer/input_layer/C17_embedding/ExpandDimsWinput_layer/sparse_input_layer/input_layer/C17_embedding/to_sparse_input/ignore_value/x*
T0*'
_output_shapes
:€€€€€€€€€
’
Pinput_layer/sparse_input_layer/input_layer/C17_embedding/to_sparse_input/indicesWhereQinput_layer/sparse_input_layer/input_layer/C17_embedding/to_sparse_input/NotEqual*'
_output_shapes
:€€€€€€€€€
ґ
Oinput_layer/sparse_input_layer/input_layer/C17_embedding/to_sparse_input/valuesGatherNdCinput_layer/sparse_input_layer/input_layer/C17_embedding/ExpandDimsPinput_layer/sparse_input_layer/input_layer/C17_embedding/to_sparse_input/indices*
Tindices0	*
Tparams0*#
_output_shapes
:€€€€€€€€€
„
Tinput_layer/sparse_input_layer/input_layer/C17_embedding/to_sparse_input/dense_shapeShapeCinput_layer/sparse_input_layer/input_layer/C17_embedding/ExpandDims*
_output_shapes
:*
T0*
out_type0	
г
?input_layer/sparse_input_layer/input_layer/C17_embedding/lookupStringToHashBucketFastOinput_layer/sparse_input_layer/input_layer/C17_embedding/to_sparse_input/values*
num_bucketsРN*#
_output_shapes
:€€€€€€€€€
Э
minput_layer/sparse_input_layer/input_layer/C17_embedding/embedding_weights/Initializer/truncated_normal/shapeConst*
_output_shapes
:*
valueB"'     *
dtype0*]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C17_embedding/embedding_weights
Р
linput_layer/sparse_input_layer/input_layer/C17_embedding/embedding_weights/Initializer/truncated_normal/meanConst*
valueB
 *    *]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C17_embedding/embedding_weights*
_output_shapes
: *
dtype0
Т
ninput_layer/sparse_input_layer/input_layer/C17_embedding/embedding_weights/Initializer/truncated_normal/stddevConst*]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C17_embedding/embedding_weights*
_output_shapes
: *
dtype0*
valueB
 *  А>
П
winput_layer/sparse_input_layer/input_layer/C17_embedding/embedding_weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormalminput_layer/sparse_input_layer/input_layer/C17_embedding/embedding_weights/Initializer/truncated_normal/shape*
dtype0*
T0*
_output_shapes
:	РN*]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C17_embedding/embedding_weights
д
kinput_layer/sparse_input_layer/input_layer/C17_embedding/embedding_weights/Initializer/truncated_normal/mulMulwinput_layer/sparse_input_layer/input_layer/C17_embedding/embedding_weights/Initializer/truncated_normal/TruncatedNormalninput_layer/sparse_input_layer/input_layer/C17_embedding/embedding_weights/Initializer/truncated_normal/stddev*
_output_shapes
:	РN*]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C17_embedding/embedding_weights*
T0
“
ginput_layer/sparse_input_layer/input_layer/C17_embedding/embedding_weights/Initializer/truncated_normalAddkinput_layer/sparse_input_layer/input_layer/C17_embedding/embedding_weights/Initializer/truncated_normal/mullinput_layer/sparse_input_layer/input_layer/C17_embedding/embedding_weights/Initializer/truncated_normal/mean*]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C17_embedding/embedding_weights*
_output_shapes
:	РN*
T0
ы
Jinput_layer/sparse_input_layer/input_layer/C17_embedding/embedding_weights
VariableV2*
dtype0*
_output_shapes
:	РN*]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C17_embedding/embedding_weights*
shape:	РN
Щ
Qinput_layer/sparse_input_layer/input_layer/C17_embedding/embedding_weights/AssignAssignJinput_layer/sparse_input_layer/input_layer/C17_embedding/embedding_weightsginput_layer/sparse_input_layer/input_layer/C17_embedding/embedding_weights/Initializer/truncated_normal*
_output_shapes
:	РN*]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C17_embedding/embedding_weights*
T0
∞
Oinput_layer/sparse_input_layer/input_layer/C17_embedding/embedding_weights/readIdentityJinput_layer/sparse_input_layer/input_layer/C17_embedding/embedding_weights*]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C17_embedding/embedding_weights*
_output_shapes
:	РN*
T0
§
Zinput_layer/sparse_input_layer/input_layer/C17_embedding/C17_embedding_weights/Slice/beginConst*
_output_shapes
:*
valueB: *
dtype0
£
Yinput_layer/sparse_input_layer/input_layer/C17_embedding/C17_embedding_weights/Slice/sizeConst*
valueB:*
dtype0*
_output_shapes
:
Ь
Tinput_layer/sparse_input_layer/input_layer/C17_embedding/C17_embedding_weights/SliceSliceTinput_layer/sparse_input_layer/input_layer/C17_embedding/to_sparse_input/dense_shapeZinput_layer/sparse_input_layer/input_layer/C17_embedding/C17_embedding_weights/Slice/beginYinput_layer/sparse_input_layer/input_layer/C17_embedding/C17_embedding_weights/Slice/size*
_output_shapes
:*
T0	*
Index0
Ю
Tinput_layer/sparse_input_layer/input_layer/C17_embedding/C17_embedding_weights/ConstConst*
valueB: *
dtype0*
_output_shapes
:
®
Sinput_layer/sparse_input_layer/input_layer/C17_embedding/C17_embedding_weights/ProdProdTinput_layer/sparse_input_layer/input_layer/C17_embedding/C17_embedding_weights/SliceTinput_layer/sparse_input_layer/input_layer/C17_embedding/C17_embedding_weights/Const*
T0	*
_output_shapes
: 
°
_input_layer/sparse_input_layer/input_layer/C17_embedding/C17_embedding_weights/GatherV2/indicesConst*
_output_shapes
: *
value	B :*
dtype0
Ю
\input_layer/sparse_input_layer/input_layer/C17_embedding/C17_embedding_weights/GatherV2/axisConst*
value	B : *
_output_shapes
: *
dtype0
Љ
Winput_layer/sparse_input_layer/input_layer/C17_embedding/C17_embedding_weights/GatherV2GatherV2Tinput_layer/sparse_input_layer/input_layer/C17_embedding/to_sparse_input/dense_shape_input_layer/sparse_input_layer/input_layer/C17_embedding/C17_embedding_weights/GatherV2/indices\input_layer/sparse_input_layer/input_layer/C17_embedding/C17_embedding_weights/GatherV2/axis*
_output_shapes
: *
Taxis0*
Tparams0	*
Tindices0
є
Uinput_layer/sparse_input_layer/input_layer/C17_embedding/C17_embedding_weights/Cast/xPackSinput_layer/sparse_input_layer/input_layer/C17_embedding/C17_embedding_weights/ProdWinput_layer/sparse_input_layer/input_layer/C17_embedding/C17_embedding_weights/GatherV2*
T0	*
N*
_output_shapes
:
Ы
\input_layer/sparse_input_layer/input_layer/C17_embedding/C17_embedding_weights/SparseReshapeSparseReshapePinput_layer/sparse_input_layer/input_layer/C17_embedding/to_sparse_input/indicesTinput_layer/sparse_input_layer/input_layer/C17_embedding/to_sparse_input/dense_shapeUinput_layer/sparse_input_layer/input_layer/C17_embedding/C17_embedding_weights/Cast/x*-
_output_shapes
:€€€€€€€€€:
а
einput_layer/sparse_input_layer/input_layer/C17_embedding/C17_embedding_weights/SparseReshape/IdentityIdentity?input_layer/sparse_input_layer/input_layer/C17_embedding/lookup*
T0	*#
_output_shapes
:€€€€€€€€€
Я
]input_layer/sparse_input_layer/input_layer/C17_embedding/C17_embedding_weights/GreaterEqual/yConst*
_output_shapes
: *
value	B	 R *
dtype0	
я
[input_layer/sparse_input_layer/input_layer/C17_embedding/C17_embedding_weights/GreaterEqualGreaterEqualeinput_layer/sparse_input_layer/input_layer/C17_embedding/C17_embedding_weights/SparseReshape/Identity]input_layer/sparse_input_layer/input_layer/C17_embedding/C17_embedding_weights/GreaterEqual/y*#
_output_shapes
:€€€€€€€€€*
T0	
г
Tinput_layer/sparse_input_layer/input_layer/C17_embedding/C17_embedding_weights/WhereWhere[input_layer/sparse_input_layer/input_layer/C17_embedding/C17_embedding_weights/GreaterEqual*'
_output_shapes
:€€€€€€€€€
ѓ
\input_layer/sparse_input_layer/input_layer/C17_embedding/C17_embedding_weights/Reshape/shapeConst*
valueB:
€€€€€€€€€*
dtype0*
_output_shapes
:
√
Vinput_layer/sparse_input_layer/input_layer/C17_embedding/C17_embedding_weights/ReshapeReshapeTinput_layer/sparse_input_layer/input_layer/C17_embedding/C17_embedding_weights/Where\input_layer/sparse_input_layer/input_layer/C17_embedding/C17_embedding_weights/Reshape/shape*
T0	*#
_output_shapes
:€€€€€€€€€
†
^input_layer/sparse_input_layer/input_layer/C17_embedding/C17_embedding_weights/GatherV2_1/axisConst*
dtype0*
_output_shapes
: *
value	B : 
–
Yinput_layer/sparse_input_layer/input_layer/C17_embedding/C17_embedding_weights/GatherV2_1GatherV2\input_layer/sparse_input_layer/input_layer/C17_embedding/C17_embedding_weights/SparseReshapeVinput_layer/sparse_input_layer/input_layer/C17_embedding/C17_embedding_weights/Reshape^input_layer/sparse_input_layer/input_layer/C17_embedding/C17_embedding_weights/GatherV2_1/axis*'
_output_shapes
:€€€€€€€€€*
Tparams0	*
Taxis0*
Tindices0	
†
^input_layer/sparse_input_layer/input_layer/C17_embedding/C17_embedding_weights/GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
value	B : 
’
Yinput_layer/sparse_input_layer/input_layer/C17_embedding/C17_embedding_weights/GatherV2_2GatherV2einput_layer/sparse_input_layer/input_layer/C17_embedding/C17_embedding_weights/SparseReshape/IdentityVinput_layer/sparse_input_layer/input_layer/C17_embedding/C17_embedding_weights/Reshape^input_layer/sparse_input_layer/input_layer/C17_embedding/C17_embedding_weights/GatherV2_2/axis*
Tindices0	*
Taxis0*
Tparams0	*#
_output_shapes
:€€€€€€€€€
и
Winput_layer/sparse_input_layer/input_layer/C17_embedding/C17_embedding_weights/IdentityIdentity^input_layer/sparse_input_layer/input_layer/C17_embedding/C17_embedding_weights/SparseReshape:1*
T0	*
_output_shapes
:
™
hinput_layer/sparse_input_layer/input_layer/C17_embedding/C17_embedding_weights/SparseFillEmptyRows/ConstConst*
dtype0	*
_output_shapes
: *
value	B	 R 
е
vinput_layer/sparse_input_layer/input_layer/C17_embedding/C17_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowsSparseFillEmptyRowsYinput_layer/sparse_input_layer/input_layer/C17_embedding/C17_embedding_weights/GatherV2_1Yinput_layer/sparse_input_layer/input_layer/C17_embedding/C17_embedding_weights/GatherV2_2Winput_layer/sparse_input_layer/input_layer/C17_embedding/C17_embedding_weights/Identityhinput_layer/sparse_input_layer/input_layer/C17_embedding/C17_embedding_weights/SparseFillEmptyRows/Const*
T0	*T
_output_shapesB
@:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€
Ћ
zinput_layer/sparse_input_layer/input_layer/C17_embedding/C17_embedding_weights/embedding_lookup_sparse/strided_slice/stackConst*
valueB"        *
_output_shapes
:*
dtype0
Ќ
|input_layer/sparse_input_layer/input_layer/C17_embedding/C17_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       
Ќ
|input_layer/sparse_input_layer/input_layer/C17_embedding/C17_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2Const*
dtype0*
valueB"      *
_output_shapes
:
й
tinput_layer/sparse_input_layer/input_layer/C17_embedding/C17_embedding_weights/embedding_lookup_sparse/strided_sliceStridedSlicevinput_layer/sparse_input_layer/input_layer/C17_embedding/C17_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowszinput_layer/sparse_input_layer/input_layer/C17_embedding/C17_embedding_weights/embedding_lookup_sparse/strided_slice/stack|input_layer/sparse_input_layer/input_layer/C17_embedding/C17_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1|input_layer/sparse_input_layer/input_layer/C17_embedding/C17_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2*
Index0*
T0	*
end_mask*
shrink_axis_mask*

begin_mask*#
_output_shapes
:€€€€€€€€€
¶
kinput_layer/sparse_input_layer/input_layer/C17_embedding/C17_embedding_weights/embedding_lookup_sparse/CastCasttinput_layer/sparse_input_layer/input_layer/C17_embedding/C17_embedding_weights/embedding_lookup_sparse/strided_slice*#
_output_shapes
:€€€€€€€€€*

SrcT0	*

DstT0
Ѓ
minput_layer/sparse_input_layer/input_layer/C17_embedding/C17_embedding_weights/embedding_lookup_sparse/UniqueUniquexinput_layer/sparse_input_layer/input_layer/C17_embedding/C17_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:1*
T0	*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
Э
|input_layer/sparse_input_layer/input_layer/C17_embedding/C17_embedding_weights/embedding_lookup_sparse/embedding_lookup/axisConst*]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C17_embedding/embedding_weights*
_output_shapes
: *
value	B : *
dtype0
х
winput_layer/sparse_input_layer/input_layer/C17_embedding/C17_embedding_weights/embedding_lookup_sparse/embedding_lookupGatherV2Oinput_layer/sparse_input_layer/input_layer/C17_embedding/embedding_weights/readminput_layer/sparse_input_layer/input_layer/C17_embedding/C17_embedding_weights/embedding_lookup_sparse/Unique|input_layer/sparse_input_layer/input_layer/C17_embedding/C17_embedding_weights/embedding_lookup_sparse/embedding_lookup/axis*
Tparams0*
Tindices0	*'
_output_shapes
:€€€€€€€€€*]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C17_embedding/embedding_weights*
Taxis0
Є
Аinput_layer/sparse_input_layer/input_layer/C17_embedding/C17_embedding_weights/embedding_lookup_sparse/embedding_lookup/IdentityIdentitywinput_layer/sparse_input_layer/input_layer/C17_embedding/C17_embedding_weights/embedding_lookup_sparse/embedding_lookup*
T0*'
_output_shapes
:€€€€€€€€€
О
finput_layer/sparse_input_layer/input_layer/C17_embedding/C17_embedding_weights/embedding_lookup_sparseSparseSegmentMeanАinput_layer/sparse_input_layer/input_layer/C17_embedding/C17_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identityoinput_layer/sparse_input_layer/input_layer/C17_embedding/C17_embedding_weights/embedding_lookup_sparse/Unique:1kinput_layer/sparse_input_layer/input_layer/C17_embedding/C17_embedding_weights/embedding_lookup_sparse/Cast*'
_output_shapes
:€€€€€€€€€*
T0
ѓ
^input_layer/sparse_input_layer/input_layer/C17_embedding/C17_embedding_weights/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   
п
Xinput_layer/sparse_input_layer/input_layer/C17_embedding/C17_embedding_weights/Reshape_1Reshapexinput_layer/sparse_input_layer/input_layer/C17_embedding/C17_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:2^input_layer/sparse_input_layer/input_layer/C17_embedding/C17_embedding_weights/Reshape_1/shape*'
_output_shapes
:€€€€€€€€€*
T0

к
Tinput_layer/sparse_input_layer/input_layer/C17_embedding/C17_embedding_weights/ShapeShapefinput_layer/sparse_input_layer/input_layer/C17_embedding/C17_embedding_weights/embedding_lookup_sparse*
T0*
_output_shapes
:
ђ
binput_layer/sparse_input_layer/input_layer/C17_embedding/C17_embedding_weights/strided_slice/stackConst*
valueB:*
_output_shapes
:*
dtype0
Ѓ
dinput_layer/sparse_input_layer/input_layer/C17_embedding/C17_embedding_weights/strided_slice/stack_1Const*
valueB:*
_output_shapes
:*
dtype0
Ѓ
dinput_layer/sparse_input_layer/input_layer/C17_embedding/C17_embedding_weights/strided_slice/stack_2Const*
_output_shapes
:*
valueB:*
dtype0
Є
\input_layer/sparse_input_layer/input_layer/C17_embedding/C17_embedding_weights/strided_sliceStridedSliceTinput_layer/sparse_input_layer/input_layer/C17_embedding/C17_embedding_weights/Shapebinput_layer/sparse_input_layer/input_layer/C17_embedding/C17_embedding_weights/strided_slice/stackdinput_layer/sparse_input_layer/input_layer/C17_embedding/C17_embedding_weights/strided_slice/stack_1dinput_layer/sparse_input_layer/input_layer/C17_embedding/C17_embedding_weights/strided_slice/stack_2*
T0*
Index0*
_output_shapes
: *
shrink_axis_mask
Ш
Vinput_layer/sparse_input_layer/input_layer/C17_embedding/C17_embedding_weights/stack/0Const*
_output_shapes
: *
value	B :*
dtype0
ј
Tinput_layer/sparse_input_layer/input_layer/C17_embedding/C17_embedding_weights/stackPackVinput_layer/sparse_input_layer/input_layer/C17_embedding/C17_embedding_weights/stack/0\input_layer/sparse_input_layer/input_layer/C17_embedding/C17_embedding_weights/strided_slice*
_output_shapes
:*
N*
T0
∆
Sinput_layer/sparse_input_layer/input_layer/C17_embedding/C17_embedding_weights/TileTileXinput_layer/sparse_input_layer/input_layer/C17_embedding/C17_embedding_weights/Reshape_1Tinput_layer/sparse_input_layer/input_layer/C17_embedding/C17_embedding_weights/stack*
T0
*0
_output_shapes
:€€€€€€€€€€€€€€€€€€
А
Yinput_layer/sparse_input_layer/input_layer/C17_embedding/C17_embedding_weights/zeros_like	ZerosLikefinput_layer/sparse_input_layer/input_layer/C17_embedding/C17_embedding_weights/embedding_lookup_sparse*'
_output_shapes
:€€€€€€€€€*
T0
Ґ
Ninput_layer/sparse_input_layer/input_layer/C17_embedding/C17_embedding_weightsSelectSinput_layer/sparse_input_layer/input_layer/C17_embedding/C17_embedding_weights/TileYinput_layer/sparse_input_layer/input_layer/C17_embedding/C17_embedding_weights/zeros_likefinput_layer/sparse_input_layer/input_layer/C17_embedding/C17_embedding_weights/embedding_lookup_sparse*'
_output_shapes
:€€€€€€€€€*
T0
з
Uinput_layer/sparse_input_layer/input_layer/C17_embedding/C17_embedding_weights/Cast_1CastTinput_layer/sparse_input_layer/input_layer/C17_embedding/to_sparse_input/dense_shape*

SrcT0	*
_output_shapes
:*

DstT0
¶
\input_layer/sparse_input_layer/input_layer/C17_embedding/C17_embedding_weights/Slice_1/beginConst*
dtype0*
valueB: *
_output_shapes
:
•
[input_layer/sparse_input_layer/input_layer/C17_embedding/C17_embedding_weights/Slice_1/sizeConst*
dtype0*
valueB:*
_output_shapes
:
£
Vinput_layer/sparse_input_layer/input_layer/C17_embedding/C17_embedding_weights/Slice_1SliceUinput_layer/sparse_input_layer/input_layer/C17_embedding/C17_embedding_weights/Cast_1\input_layer/sparse_input_layer/input_layer/C17_embedding/C17_embedding_weights/Slice_1/begin[input_layer/sparse_input_layer/input_layer/C17_embedding/C17_embedding_weights/Slice_1/size*
T0*
_output_shapes
:*
Index0
‘
Vinput_layer/sparse_input_layer/input_layer/C17_embedding/C17_embedding_weights/Shape_1ShapeNinput_layer/sparse_input_layer/input_layer/C17_embedding/C17_embedding_weights*
T0*
_output_shapes
:
¶
\input_layer/sparse_input_layer/input_layer/C17_embedding/C17_embedding_weights/Slice_2/beginConst*
valueB:*
_output_shapes
:*
dtype0
Ѓ
[input_layer/sparse_input_layer/input_layer/C17_embedding/C17_embedding_weights/Slice_2/sizeConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€
§
Vinput_layer/sparse_input_layer/input_layer/C17_embedding/C17_embedding_weights/Slice_2SliceVinput_layer/sparse_input_layer/input_layer/C17_embedding/C17_embedding_weights/Shape_1\input_layer/sparse_input_layer/input_layer/C17_embedding/C17_embedding_weights/Slice_2/begin[input_layer/sparse_input_layer/input_layer/C17_embedding/C17_embedding_weights/Slice_2/size*
T0*
Index0*
_output_shapes
:
Ь
Zinput_layer/sparse_input_layer/input_layer/C17_embedding/C17_embedding_weights/concat/axisConst*
_output_shapes
: *
value	B : *
dtype0
Ы
Uinput_layer/sparse_input_layer/input_layer/C17_embedding/C17_embedding_weights/concatConcatV2Vinput_layer/sparse_input_layer/input_layer/C17_embedding/C17_embedding_weights/Slice_1Vinput_layer/sparse_input_layer/input_layer/C17_embedding/C17_embedding_weights/Slice_2Zinput_layer/sparse_input_layer/input_layer/C17_embedding/C17_embedding_weights/concat/axis*
N*
T0*
_output_shapes
:
Љ
Xinput_layer/sparse_input_layer/input_layer/C17_embedding/C17_embedding_weights/Reshape_2ReshapeNinput_layer/sparse_input_layer/input_layer/C17_embedding/C17_embedding_weightsUinput_layer/sparse_input_layer/input_layer/C17_embedding/C17_embedding_weights/concat*
T0*'
_output_shapes
:€€€€€€€€€
∆
>input_layer/sparse_input_layer/input_layer/C17_embedding/ShapeShapeXinput_layer/sparse_input_layer/input_layer/C17_embedding/C17_embedding_weights/Reshape_2*
T0*
_output_shapes
:
Ц
Linput_layer/sparse_input_layer/input_layer/C17_embedding/strided_slice/stackConst*
dtype0*
valueB: *
_output_shapes
:
Ш
Ninput_layer/sparse_input_layer/input_layer/C17_embedding/strided_slice/stack_1Const*
dtype0*
valueB:*
_output_shapes
:
Ш
Ninput_layer/sparse_input_layer/input_layer/C17_embedding/strided_slice/stack_2Const*
_output_shapes
:*
valueB:*
dtype0
 
Finput_layer/sparse_input_layer/input_layer/C17_embedding/strided_sliceStridedSlice>input_layer/sparse_input_layer/input_layer/C17_embedding/ShapeLinput_layer/sparse_input_layer/input_layer/C17_embedding/strided_slice/stackNinput_layer/sparse_input_layer/input_layer/C17_embedding/strided_slice/stack_1Ninput_layer/sparse_input_layer/input_layer/C17_embedding/strided_slice/stack_2*
Index0*
T0*
shrink_axis_mask*
_output_shapes
: 
К
Hinput_layer/sparse_input_layer/input_layer/C17_embedding/Reshape/shape/1Const*
value	B :*
dtype0*
_output_shapes
: 
О
Finput_layer/sparse_input_layer/input_layer/C17_embedding/Reshape/shapePackFinput_layer/sparse_input_layer/input_layer/C17_embedding/strided_sliceHinput_layer/sparse_input_layer/input_layer/C17_embedding/Reshape/shape/1*
T0*
N*
_output_shapes
:
Я
@input_layer/sparse_input_layer/input_layer/C17_embedding/ReshapeReshapeXinput_layer/sparse_input_layer/input_layer/C17_embedding/C17_embedding_weights/Reshape_2Finput_layer/sparse_input_layer/input_layer/C17_embedding/Reshape/shape*'
_output_shapes
:€€€€€€€€€*
T0
Т
Ginput_layer/sparse_input_layer/input_layer/C18_embedding/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€
№
Cinput_layer/sparse_input_layer/input_layer/C18_embedding/ExpandDims
ExpandDimsPlaceholder_30Ginput_layer/sparse_input_layer/input_layer/C18_embedding/ExpandDims/dim*
T0*'
_output_shapes
:€€€€€€€€€
Ш
Winput_layer/sparse_input_layer/input_layer/C18_embedding/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB B 
≠
Qinput_layer/sparse_input_layer/input_layer/C18_embedding/to_sparse_input/NotEqualNotEqualCinput_layer/sparse_input_layer/input_layer/C18_embedding/ExpandDimsWinput_layer/sparse_input_layer/input_layer/C18_embedding/to_sparse_input/ignore_value/x*
T0*'
_output_shapes
:€€€€€€€€€
’
Pinput_layer/sparse_input_layer/input_layer/C18_embedding/to_sparse_input/indicesWhereQinput_layer/sparse_input_layer/input_layer/C18_embedding/to_sparse_input/NotEqual*'
_output_shapes
:€€€€€€€€€
ґ
Oinput_layer/sparse_input_layer/input_layer/C18_embedding/to_sparse_input/valuesGatherNdCinput_layer/sparse_input_layer/input_layer/C18_embedding/ExpandDimsPinput_layer/sparse_input_layer/input_layer/C18_embedding/to_sparse_input/indices*#
_output_shapes
:€€€€€€€€€*
Tparams0*
Tindices0	
„
Tinput_layer/sparse_input_layer/input_layer/C18_embedding/to_sparse_input/dense_shapeShapeCinput_layer/sparse_input_layer/input_layer/C18_embedding/ExpandDims*
_output_shapes
:*
T0*
out_type0	
г
?input_layer/sparse_input_layer/input_layer/C18_embedding/lookupStringToHashBucketFastOinput_layer/sparse_input_layer/input_layer/C18_embedding/to_sparse_input/values*
num_bucketsРN*#
_output_shapes
:€€€€€€€€€
Э
minput_layer/sparse_input_layer/input_layer/C18_embedding/embedding_weights/Initializer/truncated_normal/shapeConst*
dtype0*
_output_shapes
:*
valueB"'     *]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C18_embedding/embedding_weights
Р
linput_layer/sparse_input_layer/input_layer/C18_embedding/embedding_weights/Initializer/truncated_normal/meanConst*
dtype0*]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C18_embedding/embedding_weights*
valueB
 *    *
_output_shapes
: 
Т
ninput_layer/sparse_input_layer/input_layer/C18_embedding/embedding_weights/Initializer/truncated_normal/stddevConst*]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C18_embedding/embedding_weights*
dtype0*
_output_shapes
: *
valueB
 *  А>
П
winput_layer/sparse_input_layer/input_layer/C18_embedding/embedding_weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormalminput_layer/sparse_input_layer/input_layer/C18_embedding/embedding_weights/Initializer/truncated_normal/shape*
T0*
_output_shapes
:	РN*
dtype0*]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C18_embedding/embedding_weights
д
kinput_layer/sparse_input_layer/input_layer/C18_embedding/embedding_weights/Initializer/truncated_normal/mulMulwinput_layer/sparse_input_layer/input_layer/C18_embedding/embedding_weights/Initializer/truncated_normal/TruncatedNormalninput_layer/sparse_input_layer/input_layer/C18_embedding/embedding_weights/Initializer/truncated_normal/stddev*]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C18_embedding/embedding_weights*
_output_shapes
:	РN*
T0
“
ginput_layer/sparse_input_layer/input_layer/C18_embedding/embedding_weights/Initializer/truncated_normalAddkinput_layer/sparse_input_layer/input_layer/C18_embedding/embedding_weights/Initializer/truncated_normal/mullinput_layer/sparse_input_layer/input_layer/C18_embedding/embedding_weights/Initializer/truncated_normal/mean*
_output_shapes
:	РN*
T0*]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C18_embedding/embedding_weights
ы
Jinput_layer/sparse_input_layer/input_layer/C18_embedding/embedding_weights
VariableV2*
shape:	РN*]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C18_embedding/embedding_weights*
_output_shapes
:	РN*
dtype0
Щ
Qinput_layer/sparse_input_layer/input_layer/C18_embedding/embedding_weights/AssignAssignJinput_layer/sparse_input_layer/input_layer/C18_embedding/embedding_weightsginput_layer/sparse_input_layer/input_layer/C18_embedding/embedding_weights/Initializer/truncated_normal*
T0*]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C18_embedding/embedding_weights*
_output_shapes
:	РN
∞
Oinput_layer/sparse_input_layer/input_layer/C18_embedding/embedding_weights/readIdentityJinput_layer/sparse_input_layer/input_layer/C18_embedding/embedding_weights*
_output_shapes
:	РN*]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C18_embedding/embedding_weights*
T0
§
Zinput_layer/sparse_input_layer/input_layer/C18_embedding/C18_embedding_weights/Slice/beginConst*
_output_shapes
:*
valueB: *
dtype0
£
Yinput_layer/sparse_input_layer/input_layer/C18_embedding/C18_embedding_weights/Slice/sizeConst*
_output_shapes
:*
dtype0*
valueB:
Ь
Tinput_layer/sparse_input_layer/input_layer/C18_embedding/C18_embedding_weights/SliceSliceTinput_layer/sparse_input_layer/input_layer/C18_embedding/to_sparse_input/dense_shapeZinput_layer/sparse_input_layer/input_layer/C18_embedding/C18_embedding_weights/Slice/beginYinput_layer/sparse_input_layer/input_layer/C18_embedding/C18_embedding_weights/Slice/size*
_output_shapes
:*
T0	*
Index0
Ю
Tinput_layer/sparse_input_layer/input_layer/C18_embedding/C18_embedding_weights/ConstConst*
valueB: *
dtype0*
_output_shapes
:
®
Sinput_layer/sparse_input_layer/input_layer/C18_embedding/C18_embedding_weights/ProdProdTinput_layer/sparse_input_layer/input_layer/C18_embedding/C18_embedding_weights/SliceTinput_layer/sparse_input_layer/input_layer/C18_embedding/C18_embedding_weights/Const*
_output_shapes
: *
T0	
°
_input_layer/sparse_input_layer/input_layer/C18_embedding/C18_embedding_weights/GatherV2/indicesConst*
dtype0*
_output_shapes
: *
value	B :
Ю
\input_layer/sparse_input_layer/input_layer/C18_embedding/C18_embedding_weights/GatherV2/axisConst*
value	B : *
_output_shapes
: *
dtype0
Љ
Winput_layer/sparse_input_layer/input_layer/C18_embedding/C18_embedding_weights/GatherV2GatherV2Tinput_layer/sparse_input_layer/input_layer/C18_embedding/to_sparse_input/dense_shape_input_layer/sparse_input_layer/input_layer/C18_embedding/C18_embedding_weights/GatherV2/indices\input_layer/sparse_input_layer/input_layer/C18_embedding/C18_embedding_weights/GatherV2/axis*
_output_shapes
: *
Taxis0*
Tparams0	*
Tindices0
є
Uinput_layer/sparse_input_layer/input_layer/C18_embedding/C18_embedding_weights/Cast/xPackSinput_layer/sparse_input_layer/input_layer/C18_embedding/C18_embedding_weights/ProdWinput_layer/sparse_input_layer/input_layer/C18_embedding/C18_embedding_weights/GatherV2*
_output_shapes
:*
N*
T0	
Ы
\input_layer/sparse_input_layer/input_layer/C18_embedding/C18_embedding_weights/SparseReshapeSparseReshapePinput_layer/sparse_input_layer/input_layer/C18_embedding/to_sparse_input/indicesTinput_layer/sparse_input_layer/input_layer/C18_embedding/to_sparse_input/dense_shapeUinput_layer/sparse_input_layer/input_layer/C18_embedding/C18_embedding_weights/Cast/x*-
_output_shapes
:€€€€€€€€€:
а
einput_layer/sparse_input_layer/input_layer/C18_embedding/C18_embedding_weights/SparseReshape/IdentityIdentity?input_layer/sparse_input_layer/input_layer/C18_embedding/lookup*
T0	*#
_output_shapes
:€€€€€€€€€
Я
]input_layer/sparse_input_layer/input_layer/C18_embedding/C18_embedding_weights/GreaterEqual/yConst*
dtype0	*
value	B	 R *
_output_shapes
: 
я
[input_layer/sparse_input_layer/input_layer/C18_embedding/C18_embedding_weights/GreaterEqualGreaterEqualeinput_layer/sparse_input_layer/input_layer/C18_embedding/C18_embedding_weights/SparseReshape/Identity]input_layer/sparse_input_layer/input_layer/C18_embedding/C18_embedding_weights/GreaterEqual/y*#
_output_shapes
:€€€€€€€€€*
T0	
г
Tinput_layer/sparse_input_layer/input_layer/C18_embedding/C18_embedding_weights/WhereWhere[input_layer/sparse_input_layer/input_layer/C18_embedding/C18_embedding_weights/GreaterEqual*'
_output_shapes
:€€€€€€€€€
ѓ
\input_layer/sparse_input_layer/input_layer/C18_embedding/C18_embedding_weights/Reshape/shapeConst*
dtype0*
_output_shapes
:*
valueB:
€€€€€€€€€
√
Vinput_layer/sparse_input_layer/input_layer/C18_embedding/C18_embedding_weights/ReshapeReshapeTinput_layer/sparse_input_layer/input_layer/C18_embedding/C18_embedding_weights/Where\input_layer/sparse_input_layer/input_layer/C18_embedding/C18_embedding_weights/Reshape/shape*#
_output_shapes
:€€€€€€€€€*
T0	
†
^input_layer/sparse_input_layer/input_layer/C18_embedding/C18_embedding_weights/GatherV2_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
–
Yinput_layer/sparse_input_layer/input_layer/C18_embedding/C18_embedding_weights/GatherV2_1GatherV2\input_layer/sparse_input_layer/input_layer/C18_embedding/C18_embedding_weights/SparseReshapeVinput_layer/sparse_input_layer/input_layer/C18_embedding/C18_embedding_weights/Reshape^input_layer/sparse_input_layer/input_layer/C18_embedding/C18_embedding_weights/GatherV2_1/axis*
Taxis0*'
_output_shapes
:€€€€€€€€€*
Tindices0	*
Tparams0	
†
^input_layer/sparse_input_layer/input_layer/C18_embedding/C18_embedding_weights/GatherV2_2/axisConst*
dtype0*
value	B : *
_output_shapes
: 
’
Yinput_layer/sparse_input_layer/input_layer/C18_embedding/C18_embedding_weights/GatherV2_2GatherV2einput_layer/sparse_input_layer/input_layer/C18_embedding/C18_embedding_weights/SparseReshape/IdentityVinput_layer/sparse_input_layer/input_layer/C18_embedding/C18_embedding_weights/Reshape^input_layer/sparse_input_layer/input_layer/C18_embedding/C18_embedding_weights/GatherV2_2/axis*
Taxis0*
Tparams0	*
Tindices0	*#
_output_shapes
:€€€€€€€€€
и
Winput_layer/sparse_input_layer/input_layer/C18_embedding/C18_embedding_weights/IdentityIdentity^input_layer/sparse_input_layer/input_layer/C18_embedding/C18_embedding_weights/SparseReshape:1*
T0	*
_output_shapes
:
™
hinput_layer/sparse_input_layer/input_layer/C18_embedding/C18_embedding_weights/SparseFillEmptyRows/ConstConst*
dtype0	*
value	B	 R *
_output_shapes
: 
е
vinput_layer/sparse_input_layer/input_layer/C18_embedding/C18_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowsSparseFillEmptyRowsYinput_layer/sparse_input_layer/input_layer/C18_embedding/C18_embedding_weights/GatherV2_1Yinput_layer/sparse_input_layer/input_layer/C18_embedding/C18_embedding_weights/GatherV2_2Winput_layer/sparse_input_layer/input_layer/C18_embedding/C18_embedding_weights/Identityhinput_layer/sparse_input_layer/input_layer/C18_embedding/C18_embedding_weights/SparseFillEmptyRows/Const*T
_output_shapesB
@:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€*
T0	
Ћ
zinput_layer/sparse_input_layer/input_layer/C18_embedding/C18_embedding_weights/embedding_lookup_sparse/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        
Ќ
|input_layer/sparse_input_layer/input_layer/C18_embedding/C18_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1Const*
valueB"       *
dtype0*
_output_shapes
:
Ќ
|input_layer/sparse_input_layer/input_layer/C18_embedding/C18_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2Const*
dtype0*
valueB"      *
_output_shapes
:
й
tinput_layer/sparse_input_layer/input_layer/C18_embedding/C18_embedding_weights/embedding_lookup_sparse/strided_sliceStridedSlicevinput_layer/sparse_input_layer/input_layer/C18_embedding/C18_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowszinput_layer/sparse_input_layer/input_layer/C18_embedding/C18_embedding_weights/embedding_lookup_sparse/strided_slice/stack|input_layer/sparse_input_layer/input_layer/C18_embedding/C18_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1|input_layer/sparse_input_layer/input_layer/C18_embedding/C18_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2*#
_output_shapes
:€€€€€€€€€*

begin_mask*
shrink_axis_mask*
end_mask*
Index0*
T0	
¶
kinput_layer/sparse_input_layer/input_layer/C18_embedding/C18_embedding_weights/embedding_lookup_sparse/CastCasttinput_layer/sparse_input_layer/input_layer/C18_embedding/C18_embedding_weights/embedding_lookup_sparse/strided_slice*

SrcT0	*#
_output_shapes
:€€€€€€€€€*

DstT0
Ѓ
minput_layer/sparse_input_layer/input_layer/C18_embedding/C18_embedding_weights/embedding_lookup_sparse/UniqueUniquexinput_layer/sparse_input_layer/input_layer/C18_embedding/C18_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:1*
T0	*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
Э
|input_layer/sparse_input_layer/input_layer/C18_embedding/C18_embedding_weights/embedding_lookup_sparse/embedding_lookup/axisConst*
_output_shapes
: *]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C18_embedding/embedding_weights*
value	B : *
dtype0
х
winput_layer/sparse_input_layer/input_layer/C18_embedding/C18_embedding_weights/embedding_lookup_sparse/embedding_lookupGatherV2Oinput_layer/sparse_input_layer/input_layer/C18_embedding/embedding_weights/readminput_layer/sparse_input_layer/input_layer/C18_embedding/C18_embedding_weights/embedding_lookup_sparse/Unique|input_layer/sparse_input_layer/input_layer/C18_embedding/C18_embedding_weights/embedding_lookup_sparse/embedding_lookup/axis*
Taxis0*
Tindices0	*]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C18_embedding/embedding_weights*'
_output_shapes
:€€€€€€€€€*
Tparams0
Є
Аinput_layer/sparse_input_layer/input_layer/C18_embedding/C18_embedding_weights/embedding_lookup_sparse/embedding_lookup/IdentityIdentitywinput_layer/sparse_input_layer/input_layer/C18_embedding/C18_embedding_weights/embedding_lookup_sparse/embedding_lookup*'
_output_shapes
:€€€€€€€€€*
T0
О
finput_layer/sparse_input_layer/input_layer/C18_embedding/C18_embedding_weights/embedding_lookup_sparseSparseSegmentMeanАinput_layer/sparse_input_layer/input_layer/C18_embedding/C18_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identityoinput_layer/sparse_input_layer/input_layer/C18_embedding/C18_embedding_weights/embedding_lookup_sparse/Unique:1kinput_layer/sparse_input_layer/input_layer/C18_embedding/C18_embedding_weights/embedding_lookup_sparse/Cast*
T0*'
_output_shapes
:€€€€€€€€€
ѓ
^input_layer/sparse_input_layer/input_layer/C18_embedding/C18_embedding_weights/Reshape_1/shapeConst*
dtype0*
_output_shapes
:*
valueB"€€€€   
п
Xinput_layer/sparse_input_layer/input_layer/C18_embedding/C18_embedding_weights/Reshape_1Reshapexinput_layer/sparse_input_layer/input_layer/C18_embedding/C18_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:2^input_layer/sparse_input_layer/input_layer/C18_embedding/C18_embedding_weights/Reshape_1/shape*'
_output_shapes
:€€€€€€€€€*
T0

к
Tinput_layer/sparse_input_layer/input_layer/C18_embedding/C18_embedding_weights/ShapeShapefinput_layer/sparse_input_layer/input_layer/C18_embedding/C18_embedding_weights/embedding_lookup_sparse*
_output_shapes
:*
T0
ђ
binput_layer/sparse_input_layer/input_layer/C18_embedding/C18_embedding_weights/strided_slice/stackConst*
valueB:*
dtype0*
_output_shapes
:
Ѓ
dinput_layer/sparse_input_layer/input_layer/C18_embedding/C18_embedding_weights/strided_slice/stack_1Const*
dtype0*
valueB:*
_output_shapes
:
Ѓ
dinput_layer/sparse_input_layer/input_layer/C18_embedding/C18_embedding_weights/strided_slice/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
Є
\input_layer/sparse_input_layer/input_layer/C18_embedding/C18_embedding_weights/strided_sliceStridedSliceTinput_layer/sparse_input_layer/input_layer/C18_embedding/C18_embedding_weights/Shapebinput_layer/sparse_input_layer/input_layer/C18_embedding/C18_embedding_weights/strided_slice/stackdinput_layer/sparse_input_layer/input_layer/C18_embedding/C18_embedding_weights/strided_slice/stack_1dinput_layer/sparse_input_layer/input_layer/C18_embedding/C18_embedding_weights/strided_slice/stack_2*
Index0*
T0*
shrink_axis_mask*
_output_shapes
: 
Ш
Vinput_layer/sparse_input_layer/input_layer/C18_embedding/C18_embedding_weights/stack/0Const*
dtype0*
value	B :*
_output_shapes
: 
ј
Tinput_layer/sparse_input_layer/input_layer/C18_embedding/C18_embedding_weights/stackPackVinput_layer/sparse_input_layer/input_layer/C18_embedding/C18_embedding_weights/stack/0\input_layer/sparse_input_layer/input_layer/C18_embedding/C18_embedding_weights/strided_slice*
T0*
N*
_output_shapes
:
∆
Sinput_layer/sparse_input_layer/input_layer/C18_embedding/C18_embedding_weights/TileTileXinput_layer/sparse_input_layer/input_layer/C18_embedding/C18_embedding_weights/Reshape_1Tinput_layer/sparse_input_layer/input_layer/C18_embedding/C18_embedding_weights/stack*0
_output_shapes
:€€€€€€€€€€€€€€€€€€*
T0

А
Yinput_layer/sparse_input_layer/input_layer/C18_embedding/C18_embedding_weights/zeros_like	ZerosLikefinput_layer/sparse_input_layer/input_layer/C18_embedding/C18_embedding_weights/embedding_lookup_sparse*
T0*'
_output_shapes
:€€€€€€€€€
Ґ
Ninput_layer/sparse_input_layer/input_layer/C18_embedding/C18_embedding_weightsSelectSinput_layer/sparse_input_layer/input_layer/C18_embedding/C18_embedding_weights/TileYinput_layer/sparse_input_layer/input_layer/C18_embedding/C18_embedding_weights/zeros_likefinput_layer/sparse_input_layer/input_layer/C18_embedding/C18_embedding_weights/embedding_lookup_sparse*
T0*'
_output_shapes
:€€€€€€€€€
з
Uinput_layer/sparse_input_layer/input_layer/C18_embedding/C18_embedding_weights/Cast_1CastTinput_layer/sparse_input_layer/input_layer/C18_embedding/to_sparse_input/dense_shape*
_output_shapes
:*

DstT0*

SrcT0	
¶
\input_layer/sparse_input_layer/input_layer/C18_embedding/C18_embedding_weights/Slice_1/beginConst*
valueB: *
dtype0*
_output_shapes
:
•
[input_layer/sparse_input_layer/input_layer/C18_embedding/C18_embedding_weights/Slice_1/sizeConst*
_output_shapes
:*
valueB:*
dtype0
£
Vinput_layer/sparse_input_layer/input_layer/C18_embedding/C18_embedding_weights/Slice_1SliceUinput_layer/sparse_input_layer/input_layer/C18_embedding/C18_embedding_weights/Cast_1\input_layer/sparse_input_layer/input_layer/C18_embedding/C18_embedding_weights/Slice_1/begin[input_layer/sparse_input_layer/input_layer/C18_embedding/C18_embedding_weights/Slice_1/size*
T0*
Index0*
_output_shapes
:
‘
Vinput_layer/sparse_input_layer/input_layer/C18_embedding/C18_embedding_weights/Shape_1ShapeNinput_layer/sparse_input_layer/input_layer/C18_embedding/C18_embedding_weights*
T0*
_output_shapes
:
¶
\input_layer/sparse_input_layer/input_layer/C18_embedding/C18_embedding_weights/Slice_2/beginConst*
dtype0*
_output_shapes
:*
valueB:
Ѓ
[input_layer/sparse_input_layer/input_layer/C18_embedding/C18_embedding_weights/Slice_2/sizeConst*
valueB:
€€€€€€€€€*
dtype0*
_output_shapes
:
§
Vinput_layer/sparse_input_layer/input_layer/C18_embedding/C18_embedding_weights/Slice_2SliceVinput_layer/sparse_input_layer/input_layer/C18_embedding/C18_embedding_weights/Shape_1\input_layer/sparse_input_layer/input_layer/C18_embedding/C18_embedding_weights/Slice_2/begin[input_layer/sparse_input_layer/input_layer/C18_embedding/C18_embedding_weights/Slice_2/size*
T0*
Index0*
_output_shapes
:
Ь
Zinput_layer/sparse_input_layer/input_layer/C18_embedding/C18_embedding_weights/concat/axisConst*
dtype0*
value	B : *
_output_shapes
: 
Ы
Uinput_layer/sparse_input_layer/input_layer/C18_embedding/C18_embedding_weights/concatConcatV2Vinput_layer/sparse_input_layer/input_layer/C18_embedding/C18_embedding_weights/Slice_1Vinput_layer/sparse_input_layer/input_layer/C18_embedding/C18_embedding_weights/Slice_2Zinput_layer/sparse_input_layer/input_layer/C18_embedding/C18_embedding_weights/concat/axis*
N*
T0*
_output_shapes
:
Љ
Xinput_layer/sparse_input_layer/input_layer/C18_embedding/C18_embedding_weights/Reshape_2ReshapeNinput_layer/sparse_input_layer/input_layer/C18_embedding/C18_embedding_weightsUinput_layer/sparse_input_layer/input_layer/C18_embedding/C18_embedding_weights/concat*'
_output_shapes
:€€€€€€€€€*
T0
∆
>input_layer/sparse_input_layer/input_layer/C18_embedding/ShapeShapeXinput_layer/sparse_input_layer/input_layer/C18_embedding/C18_embedding_weights/Reshape_2*
T0*
_output_shapes
:
Ц
Linput_layer/sparse_input_layer/input_layer/C18_embedding/strided_slice/stackConst*
valueB: *
_output_shapes
:*
dtype0
Ш
Ninput_layer/sparse_input_layer/input_layer/C18_embedding/strided_slice/stack_1Const*
dtype0*
valueB:*
_output_shapes
:
Ш
Ninput_layer/sparse_input_layer/input_layer/C18_embedding/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
 
Finput_layer/sparse_input_layer/input_layer/C18_embedding/strided_sliceStridedSlice>input_layer/sparse_input_layer/input_layer/C18_embedding/ShapeLinput_layer/sparse_input_layer/input_layer/C18_embedding/strided_slice/stackNinput_layer/sparse_input_layer/input_layer/C18_embedding/strided_slice/stack_1Ninput_layer/sparse_input_layer/input_layer/C18_embedding/strided_slice/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
К
Hinput_layer/sparse_input_layer/input_layer/C18_embedding/Reshape/shape/1Const*
value	B :*
_output_shapes
: *
dtype0
О
Finput_layer/sparse_input_layer/input_layer/C18_embedding/Reshape/shapePackFinput_layer/sparse_input_layer/input_layer/C18_embedding/strided_sliceHinput_layer/sparse_input_layer/input_layer/C18_embedding/Reshape/shape/1*
_output_shapes
:*
T0*
N
Я
@input_layer/sparse_input_layer/input_layer/C18_embedding/ReshapeReshapeXinput_layer/sparse_input_layer/input_layer/C18_embedding/C18_embedding_weights/Reshape_2Finput_layer/sparse_input_layer/input_layer/C18_embedding/Reshape/shape*'
_output_shapes
:€€€€€€€€€*
T0
Т
Ginput_layer/sparse_input_layer/input_layer/C19_embedding/ExpandDims/dimConst*
_output_shapes
: *
valueB :
€€€€€€€€€*
dtype0
№
Cinput_layer/sparse_input_layer/input_layer/C19_embedding/ExpandDims
ExpandDimsPlaceholder_31Ginput_layer/sparse_input_layer/input_layer/C19_embedding/ExpandDims/dim*'
_output_shapes
:€€€€€€€€€*
T0
Ш
Winput_layer/sparse_input_layer/input_layer/C19_embedding/to_sparse_input/ignore_value/xConst*
dtype0*
valueB B *
_output_shapes
: 
≠
Qinput_layer/sparse_input_layer/input_layer/C19_embedding/to_sparse_input/NotEqualNotEqualCinput_layer/sparse_input_layer/input_layer/C19_embedding/ExpandDimsWinput_layer/sparse_input_layer/input_layer/C19_embedding/to_sparse_input/ignore_value/x*'
_output_shapes
:€€€€€€€€€*
T0
’
Pinput_layer/sparse_input_layer/input_layer/C19_embedding/to_sparse_input/indicesWhereQinput_layer/sparse_input_layer/input_layer/C19_embedding/to_sparse_input/NotEqual*'
_output_shapes
:€€€€€€€€€
ґ
Oinput_layer/sparse_input_layer/input_layer/C19_embedding/to_sparse_input/valuesGatherNdCinput_layer/sparse_input_layer/input_layer/C19_embedding/ExpandDimsPinput_layer/sparse_input_layer/input_layer/C19_embedding/to_sparse_input/indices*#
_output_shapes
:€€€€€€€€€*
Tindices0	*
Tparams0
„
Tinput_layer/sparse_input_layer/input_layer/C19_embedding/to_sparse_input/dense_shapeShapeCinput_layer/sparse_input_layer/input_layer/C19_embedding/ExpandDims*
out_type0	*
T0*
_output_shapes
:
г
?input_layer/sparse_input_layer/input_layer/C19_embedding/lookupStringToHashBucketFastOinput_layer/sparse_input_layer/input_layer/C19_embedding/to_sparse_input/values*
num_bucketsРN*#
_output_shapes
:€€€€€€€€€
Э
minput_layer/sparse_input_layer/input_layer/C19_embedding/embedding_weights/Initializer/truncated_normal/shapeConst*
dtype0*
valueB"'     *]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C19_embedding/embedding_weights*
_output_shapes
:
Р
linput_layer/sparse_input_layer/input_layer/C19_embedding/embedding_weights/Initializer/truncated_normal/meanConst*
valueB
 *    *
_output_shapes
: *
dtype0*]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C19_embedding/embedding_weights
Т
ninput_layer/sparse_input_layer/input_layer/C19_embedding/embedding_weights/Initializer/truncated_normal/stddevConst*]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C19_embedding/embedding_weights*
dtype0*
valueB
 *  А>*
_output_shapes
: 
П
winput_layer/sparse_input_layer/input_layer/C19_embedding/embedding_weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormalminput_layer/sparse_input_layer/input_layer/C19_embedding/embedding_weights/Initializer/truncated_normal/shape*
T0*
_output_shapes
:	РN*]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C19_embedding/embedding_weights*
dtype0
д
kinput_layer/sparse_input_layer/input_layer/C19_embedding/embedding_weights/Initializer/truncated_normal/mulMulwinput_layer/sparse_input_layer/input_layer/C19_embedding/embedding_weights/Initializer/truncated_normal/TruncatedNormalninput_layer/sparse_input_layer/input_layer/C19_embedding/embedding_weights/Initializer/truncated_normal/stddev*
T0*
_output_shapes
:	РN*]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C19_embedding/embedding_weights
“
ginput_layer/sparse_input_layer/input_layer/C19_embedding/embedding_weights/Initializer/truncated_normalAddkinput_layer/sparse_input_layer/input_layer/C19_embedding/embedding_weights/Initializer/truncated_normal/mullinput_layer/sparse_input_layer/input_layer/C19_embedding/embedding_weights/Initializer/truncated_normal/mean*]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C19_embedding/embedding_weights*
_output_shapes
:	РN*
T0
ы
Jinput_layer/sparse_input_layer/input_layer/C19_embedding/embedding_weights
VariableV2*
dtype0*
shape:	РN*
_output_shapes
:	РN*]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C19_embedding/embedding_weights
Щ
Qinput_layer/sparse_input_layer/input_layer/C19_embedding/embedding_weights/AssignAssignJinput_layer/sparse_input_layer/input_layer/C19_embedding/embedding_weightsginput_layer/sparse_input_layer/input_layer/C19_embedding/embedding_weights/Initializer/truncated_normal*
T0*]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C19_embedding/embedding_weights*
_output_shapes
:	РN
∞
Oinput_layer/sparse_input_layer/input_layer/C19_embedding/embedding_weights/readIdentityJinput_layer/sparse_input_layer/input_layer/C19_embedding/embedding_weights*
T0*
_output_shapes
:	РN*]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C19_embedding/embedding_weights
§
Zinput_layer/sparse_input_layer/input_layer/C19_embedding/C19_embedding_weights/Slice/beginConst*
_output_shapes
:*
valueB: *
dtype0
£
Yinput_layer/sparse_input_layer/input_layer/C19_embedding/C19_embedding_weights/Slice/sizeConst*
_output_shapes
:*
dtype0*
valueB:
Ь
Tinput_layer/sparse_input_layer/input_layer/C19_embedding/C19_embedding_weights/SliceSliceTinput_layer/sparse_input_layer/input_layer/C19_embedding/to_sparse_input/dense_shapeZinput_layer/sparse_input_layer/input_layer/C19_embedding/C19_embedding_weights/Slice/beginYinput_layer/sparse_input_layer/input_layer/C19_embedding/C19_embedding_weights/Slice/size*
_output_shapes
:*
Index0*
T0	
Ю
Tinput_layer/sparse_input_layer/input_layer/C19_embedding/C19_embedding_weights/ConstConst*
dtype0*
_output_shapes
:*
valueB: 
®
Sinput_layer/sparse_input_layer/input_layer/C19_embedding/C19_embedding_weights/ProdProdTinput_layer/sparse_input_layer/input_layer/C19_embedding/C19_embedding_weights/SliceTinput_layer/sparse_input_layer/input_layer/C19_embedding/C19_embedding_weights/Const*
T0	*
_output_shapes
: 
°
_input_layer/sparse_input_layer/input_layer/C19_embedding/C19_embedding_weights/GatherV2/indicesConst*
_output_shapes
: *
value	B :*
dtype0
Ю
\input_layer/sparse_input_layer/input_layer/C19_embedding/C19_embedding_weights/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 
Љ
Winput_layer/sparse_input_layer/input_layer/C19_embedding/C19_embedding_weights/GatherV2GatherV2Tinput_layer/sparse_input_layer/input_layer/C19_embedding/to_sparse_input/dense_shape_input_layer/sparse_input_layer/input_layer/C19_embedding/C19_embedding_weights/GatherV2/indices\input_layer/sparse_input_layer/input_layer/C19_embedding/C19_embedding_weights/GatherV2/axis*
Tindices0*
_output_shapes
: *
Taxis0*
Tparams0	
є
Uinput_layer/sparse_input_layer/input_layer/C19_embedding/C19_embedding_weights/Cast/xPackSinput_layer/sparse_input_layer/input_layer/C19_embedding/C19_embedding_weights/ProdWinput_layer/sparse_input_layer/input_layer/C19_embedding/C19_embedding_weights/GatherV2*
N*
T0	*
_output_shapes
:
Ы
\input_layer/sparse_input_layer/input_layer/C19_embedding/C19_embedding_weights/SparseReshapeSparseReshapePinput_layer/sparse_input_layer/input_layer/C19_embedding/to_sparse_input/indicesTinput_layer/sparse_input_layer/input_layer/C19_embedding/to_sparse_input/dense_shapeUinput_layer/sparse_input_layer/input_layer/C19_embedding/C19_embedding_weights/Cast/x*-
_output_shapes
:€€€€€€€€€:
а
einput_layer/sparse_input_layer/input_layer/C19_embedding/C19_embedding_weights/SparseReshape/IdentityIdentity?input_layer/sparse_input_layer/input_layer/C19_embedding/lookup*
T0	*#
_output_shapes
:€€€€€€€€€
Я
]input_layer/sparse_input_layer/input_layer/C19_embedding/C19_embedding_weights/GreaterEqual/yConst*
_output_shapes
: *
value	B	 R *
dtype0	
я
[input_layer/sparse_input_layer/input_layer/C19_embedding/C19_embedding_weights/GreaterEqualGreaterEqualeinput_layer/sparse_input_layer/input_layer/C19_embedding/C19_embedding_weights/SparseReshape/Identity]input_layer/sparse_input_layer/input_layer/C19_embedding/C19_embedding_weights/GreaterEqual/y*#
_output_shapes
:€€€€€€€€€*
T0	
г
Tinput_layer/sparse_input_layer/input_layer/C19_embedding/C19_embedding_weights/WhereWhere[input_layer/sparse_input_layer/input_layer/C19_embedding/C19_embedding_weights/GreaterEqual*'
_output_shapes
:€€€€€€€€€
ѓ
\input_layer/sparse_input_layer/input_layer/C19_embedding/C19_embedding_weights/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€
√
Vinput_layer/sparse_input_layer/input_layer/C19_embedding/C19_embedding_weights/ReshapeReshapeTinput_layer/sparse_input_layer/input_layer/C19_embedding/C19_embedding_weights/Where\input_layer/sparse_input_layer/input_layer/C19_embedding/C19_embedding_weights/Reshape/shape*#
_output_shapes
:€€€€€€€€€*
T0	
†
^input_layer/sparse_input_layer/input_layer/C19_embedding/C19_embedding_weights/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
–
Yinput_layer/sparse_input_layer/input_layer/C19_embedding/C19_embedding_weights/GatherV2_1GatherV2\input_layer/sparse_input_layer/input_layer/C19_embedding/C19_embedding_weights/SparseReshapeVinput_layer/sparse_input_layer/input_layer/C19_embedding/C19_embedding_weights/Reshape^input_layer/sparse_input_layer/input_layer/C19_embedding/C19_embedding_weights/GatherV2_1/axis*'
_output_shapes
:€€€€€€€€€*
Taxis0*
Tparams0	*
Tindices0	
†
^input_layer/sparse_input_layer/input_layer/C19_embedding/C19_embedding_weights/GatherV2_2/axisConst*
_output_shapes
: *
value	B : *
dtype0
’
Yinput_layer/sparse_input_layer/input_layer/C19_embedding/C19_embedding_weights/GatherV2_2GatherV2einput_layer/sparse_input_layer/input_layer/C19_embedding/C19_embedding_weights/SparseReshape/IdentityVinput_layer/sparse_input_layer/input_layer/C19_embedding/C19_embedding_weights/Reshape^input_layer/sparse_input_layer/input_layer/C19_embedding/C19_embedding_weights/GatherV2_2/axis*
Tindices0	*
Taxis0*#
_output_shapes
:€€€€€€€€€*
Tparams0	
и
Winput_layer/sparse_input_layer/input_layer/C19_embedding/C19_embedding_weights/IdentityIdentity^input_layer/sparse_input_layer/input_layer/C19_embedding/C19_embedding_weights/SparseReshape:1*
T0	*
_output_shapes
:
™
hinput_layer/sparse_input_layer/input_layer/C19_embedding/C19_embedding_weights/SparseFillEmptyRows/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R 
е
vinput_layer/sparse_input_layer/input_layer/C19_embedding/C19_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowsSparseFillEmptyRowsYinput_layer/sparse_input_layer/input_layer/C19_embedding/C19_embedding_weights/GatherV2_1Yinput_layer/sparse_input_layer/input_layer/C19_embedding/C19_embedding_weights/GatherV2_2Winput_layer/sparse_input_layer/input_layer/C19_embedding/C19_embedding_weights/Identityhinput_layer/sparse_input_layer/input_layer/C19_embedding/C19_embedding_weights/SparseFillEmptyRows/Const*T
_output_shapesB
@:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€*
T0	
Ћ
zinput_layer/sparse_input_layer/input_layer/C19_embedding/C19_embedding_weights/embedding_lookup_sparse/strided_slice/stackConst*
valueB"        *
dtype0*
_output_shapes
:
Ќ
|input_layer/sparse_input_layer/input_layer/C19_embedding/C19_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       
Ќ
|input_layer/sparse_input_layer/input_layer/C19_embedding/C19_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2Const*
_output_shapes
:*
valueB"      *
dtype0
й
tinput_layer/sparse_input_layer/input_layer/C19_embedding/C19_embedding_weights/embedding_lookup_sparse/strided_sliceStridedSlicevinput_layer/sparse_input_layer/input_layer/C19_embedding/C19_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowszinput_layer/sparse_input_layer/input_layer/C19_embedding/C19_embedding_weights/embedding_lookup_sparse/strided_slice/stack|input_layer/sparse_input_layer/input_layer/C19_embedding/C19_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1|input_layer/sparse_input_layer/input_layer/C19_embedding/C19_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2*#
_output_shapes
:€€€€€€€€€*
end_mask*
Index0*
T0	*
shrink_axis_mask*

begin_mask
¶
kinput_layer/sparse_input_layer/input_layer/C19_embedding/C19_embedding_weights/embedding_lookup_sparse/CastCasttinput_layer/sparse_input_layer/input_layer/C19_embedding/C19_embedding_weights/embedding_lookup_sparse/strided_slice*#
_output_shapes
:€€€€€€€€€*

DstT0*

SrcT0	
Ѓ
minput_layer/sparse_input_layer/input_layer/C19_embedding/C19_embedding_weights/embedding_lookup_sparse/UniqueUniquexinput_layer/sparse_input_layer/input_layer/C19_embedding/C19_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:1*
T0	*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
Э
|input_layer/sparse_input_layer/input_layer/C19_embedding/C19_embedding_weights/embedding_lookup_sparse/embedding_lookup/axisConst*
value	B : *
_output_shapes
: *
dtype0*]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C19_embedding/embedding_weights
х
winput_layer/sparse_input_layer/input_layer/C19_embedding/C19_embedding_weights/embedding_lookup_sparse/embedding_lookupGatherV2Oinput_layer/sparse_input_layer/input_layer/C19_embedding/embedding_weights/readminput_layer/sparse_input_layer/input_layer/C19_embedding/C19_embedding_weights/embedding_lookup_sparse/Unique|input_layer/sparse_input_layer/input_layer/C19_embedding/C19_embedding_weights/embedding_lookup_sparse/embedding_lookup/axis*'
_output_shapes
:€€€€€€€€€*
Taxis0*
Tindices0	*
Tparams0*]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C19_embedding/embedding_weights
Є
Аinput_layer/sparse_input_layer/input_layer/C19_embedding/C19_embedding_weights/embedding_lookup_sparse/embedding_lookup/IdentityIdentitywinput_layer/sparse_input_layer/input_layer/C19_embedding/C19_embedding_weights/embedding_lookup_sparse/embedding_lookup*'
_output_shapes
:€€€€€€€€€*
T0
О
finput_layer/sparse_input_layer/input_layer/C19_embedding/C19_embedding_weights/embedding_lookup_sparseSparseSegmentMeanАinput_layer/sparse_input_layer/input_layer/C19_embedding/C19_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identityoinput_layer/sparse_input_layer/input_layer/C19_embedding/C19_embedding_weights/embedding_lookup_sparse/Unique:1kinput_layer/sparse_input_layer/input_layer/C19_embedding/C19_embedding_weights/embedding_lookup_sparse/Cast*
T0*'
_output_shapes
:€€€€€€€€€
ѓ
^input_layer/sparse_input_layer/input_layer/C19_embedding/C19_embedding_weights/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   
п
Xinput_layer/sparse_input_layer/input_layer/C19_embedding/C19_embedding_weights/Reshape_1Reshapexinput_layer/sparse_input_layer/input_layer/C19_embedding/C19_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:2^input_layer/sparse_input_layer/input_layer/C19_embedding/C19_embedding_weights/Reshape_1/shape*'
_output_shapes
:€€€€€€€€€*
T0

к
Tinput_layer/sparse_input_layer/input_layer/C19_embedding/C19_embedding_weights/ShapeShapefinput_layer/sparse_input_layer/input_layer/C19_embedding/C19_embedding_weights/embedding_lookup_sparse*
T0*
_output_shapes
:
ђ
binput_layer/sparse_input_layer/input_layer/C19_embedding/C19_embedding_weights/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
Ѓ
dinput_layer/sparse_input_layer/input_layer/C19_embedding/C19_embedding_weights/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
Ѓ
dinput_layer/sparse_input_layer/input_layer/C19_embedding/C19_embedding_weights/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
Є
\input_layer/sparse_input_layer/input_layer/C19_embedding/C19_embedding_weights/strided_sliceStridedSliceTinput_layer/sparse_input_layer/input_layer/C19_embedding/C19_embedding_weights/Shapebinput_layer/sparse_input_layer/input_layer/C19_embedding/C19_embedding_weights/strided_slice/stackdinput_layer/sparse_input_layer/input_layer/C19_embedding/C19_embedding_weights/strided_slice/stack_1dinput_layer/sparse_input_layer/input_layer/C19_embedding/C19_embedding_weights/strided_slice/stack_2*
T0*
_output_shapes
: *
Index0*
shrink_axis_mask
Ш
Vinput_layer/sparse_input_layer/input_layer/C19_embedding/C19_embedding_weights/stack/0Const*
value	B :*
dtype0*
_output_shapes
: 
ј
Tinput_layer/sparse_input_layer/input_layer/C19_embedding/C19_embedding_weights/stackPackVinput_layer/sparse_input_layer/input_layer/C19_embedding/C19_embedding_weights/stack/0\input_layer/sparse_input_layer/input_layer/C19_embedding/C19_embedding_weights/strided_slice*
T0*
_output_shapes
:*
N
∆
Sinput_layer/sparse_input_layer/input_layer/C19_embedding/C19_embedding_weights/TileTileXinput_layer/sparse_input_layer/input_layer/C19_embedding/C19_embedding_weights/Reshape_1Tinput_layer/sparse_input_layer/input_layer/C19_embedding/C19_embedding_weights/stack*0
_output_shapes
:€€€€€€€€€€€€€€€€€€*
T0

А
Yinput_layer/sparse_input_layer/input_layer/C19_embedding/C19_embedding_weights/zeros_like	ZerosLikefinput_layer/sparse_input_layer/input_layer/C19_embedding/C19_embedding_weights/embedding_lookup_sparse*
T0*'
_output_shapes
:€€€€€€€€€
Ґ
Ninput_layer/sparse_input_layer/input_layer/C19_embedding/C19_embedding_weightsSelectSinput_layer/sparse_input_layer/input_layer/C19_embedding/C19_embedding_weights/TileYinput_layer/sparse_input_layer/input_layer/C19_embedding/C19_embedding_weights/zeros_likefinput_layer/sparse_input_layer/input_layer/C19_embedding/C19_embedding_weights/embedding_lookup_sparse*
T0*'
_output_shapes
:€€€€€€€€€
з
Uinput_layer/sparse_input_layer/input_layer/C19_embedding/C19_embedding_weights/Cast_1CastTinput_layer/sparse_input_layer/input_layer/C19_embedding/to_sparse_input/dense_shape*

SrcT0	*
_output_shapes
:*

DstT0
¶
\input_layer/sparse_input_layer/input_layer/C19_embedding/C19_embedding_weights/Slice_1/beginConst*
dtype0*
valueB: *
_output_shapes
:
•
[input_layer/sparse_input_layer/input_layer/C19_embedding/C19_embedding_weights/Slice_1/sizeConst*
_output_shapes
:*
dtype0*
valueB:
£
Vinput_layer/sparse_input_layer/input_layer/C19_embedding/C19_embedding_weights/Slice_1SliceUinput_layer/sparse_input_layer/input_layer/C19_embedding/C19_embedding_weights/Cast_1\input_layer/sparse_input_layer/input_layer/C19_embedding/C19_embedding_weights/Slice_1/begin[input_layer/sparse_input_layer/input_layer/C19_embedding/C19_embedding_weights/Slice_1/size*
_output_shapes
:*
T0*
Index0
‘
Vinput_layer/sparse_input_layer/input_layer/C19_embedding/C19_embedding_weights/Shape_1ShapeNinput_layer/sparse_input_layer/input_layer/C19_embedding/C19_embedding_weights*
T0*
_output_shapes
:
¶
\input_layer/sparse_input_layer/input_layer/C19_embedding/C19_embedding_weights/Slice_2/beginConst*
valueB:*
dtype0*
_output_shapes
:
Ѓ
[input_layer/sparse_input_layer/input_layer/C19_embedding/C19_embedding_weights/Slice_2/sizeConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€
§
Vinput_layer/sparse_input_layer/input_layer/C19_embedding/C19_embedding_weights/Slice_2SliceVinput_layer/sparse_input_layer/input_layer/C19_embedding/C19_embedding_weights/Shape_1\input_layer/sparse_input_layer/input_layer/C19_embedding/C19_embedding_weights/Slice_2/begin[input_layer/sparse_input_layer/input_layer/C19_embedding/C19_embedding_weights/Slice_2/size*
T0*
_output_shapes
:*
Index0
Ь
Zinput_layer/sparse_input_layer/input_layer/C19_embedding/C19_embedding_weights/concat/axisConst*
_output_shapes
: *
value	B : *
dtype0
Ы
Uinput_layer/sparse_input_layer/input_layer/C19_embedding/C19_embedding_weights/concatConcatV2Vinput_layer/sparse_input_layer/input_layer/C19_embedding/C19_embedding_weights/Slice_1Vinput_layer/sparse_input_layer/input_layer/C19_embedding/C19_embedding_weights/Slice_2Zinput_layer/sparse_input_layer/input_layer/C19_embedding/C19_embedding_weights/concat/axis*
N*
T0*
_output_shapes
:
Љ
Xinput_layer/sparse_input_layer/input_layer/C19_embedding/C19_embedding_weights/Reshape_2ReshapeNinput_layer/sparse_input_layer/input_layer/C19_embedding/C19_embedding_weightsUinput_layer/sparse_input_layer/input_layer/C19_embedding/C19_embedding_weights/concat*'
_output_shapes
:€€€€€€€€€*
T0
∆
>input_layer/sparse_input_layer/input_layer/C19_embedding/ShapeShapeXinput_layer/sparse_input_layer/input_layer/C19_embedding/C19_embedding_weights/Reshape_2*
_output_shapes
:*
T0
Ц
Linput_layer/sparse_input_layer/input_layer/C19_embedding/strided_slice/stackConst*
valueB: *
_output_shapes
:*
dtype0
Ш
Ninput_layer/sparse_input_layer/input_layer/C19_embedding/strided_slice/stack_1Const*
dtype0*
valueB:*
_output_shapes
:
Ш
Ninput_layer/sparse_input_layer/input_layer/C19_embedding/strided_slice/stack_2Const*
valueB:*
_output_shapes
:*
dtype0
 
Finput_layer/sparse_input_layer/input_layer/C19_embedding/strided_sliceStridedSlice>input_layer/sparse_input_layer/input_layer/C19_embedding/ShapeLinput_layer/sparse_input_layer/input_layer/C19_embedding/strided_slice/stackNinput_layer/sparse_input_layer/input_layer/C19_embedding/strided_slice/stack_1Ninput_layer/sparse_input_layer/input_layer/C19_embedding/strided_slice/stack_2*
_output_shapes
: *
Index0*
T0*
shrink_axis_mask
К
Hinput_layer/sparse_input_layer/input_layer/C19_embedding/Reshape/shape/1Const*
_output_shapes
: *
value	B :*
dtype0
О
Finput_layer/sparse_input_layer/input_layer/C19_embedding/Reshape/shapePackFinput_layer/sparse_input_layer/input_layer/C19_embedding/strided_sliceHinput_layer/sparse_input_layer/input_layer/C19_embedding/Reshape/shape/1*
N*
_output_shapes
:*
T0
Я
@input_layer/sparse_input_layer/input_layer/C19_embedding/ReshapeReshapeXinput_layer/sparse_input_layer/input_layer/C19_embedding/C19_embedding_weights/Reshape_2Finput_layer/sparse_input_layer/input_layer/C19_embedding/Reshape/shape*'
_output_shapes
:€€€€€€€€€*
T0
С
Finput_layer/sparse_input_layer/input_layer/C1_embedding/ExpandDims/dimConst*
dtype0*
_output_shapes
: *
valueB :
€€€€€€€€€
Џ
Binput_layer/sparse_input_layer/input_layer/C1_embedding/ExpandDims
ExpandDimsPlaceholder_13Finput_layer/sparse_input_layer/input_layer/C1_embedding/ExpandDims/dim*'
_output_shapes
:€€€€€€€€€*
T0
Ч
Vinput_layer/sparse_input_layer/input_layer/C1_embedding/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
valueB B *
dtype0
™
Pinput_layer/sparse_input_layer/input_layer/C1_embedding/to_sparse_input/NotEqualNotEqualBinput_layer/sparse_input_layer/input_layer/C1_embedding/ExpandDimsVinput_layer/sparse_input_layer/input_layer/C1_embedding/to_sparse_input/ignore_value/x*
T0*'
_output_shapes
:€€€€€€€€€
”
Oinput_layer/sparse_input_layer/input_layer/C1_embedding/to_sparse_input/indicesWherePinput_layer/sparse_input_layer/input_layer/C1_embedding/to_sparse_input/NotEqual*'
_output_shapes
:€€€€€€€€€
≥
Ninput_layer/sparse_input_layer/input_layer/C1_embedding/to_sparse_input/valuesGatherNdBinput_layer/sparse_input_layer/input_layer/C1_embedding/ExpandDimsOinput_layer/sparse_input_layer/input_layer/C1_embedding/to_sparse_input/indices*
Tindices0	*#
_output_shapes
:€€€€€€€€€*
Tparams0
’
Sinput_layer/sparse_input_layer/input_layer/C1_embedding/to_sparse_input/dense_shapeShapeBinput_layer/sparse_input_layer/input_layer/C1_embedding/ExpandDims*
out_type0	*
_output_shapes
:*
T0
б
>input_layer/sparse_input_layer/input_layer/C1_embedding/lookupStringToHashBucketFastNinput_layer/sparse_input_layer/input_layer/C1_embedding/to_sparse_input/values*
num_bucketsРN*#
_output_shapes
:€€€€€€€€€
Ы
linput_layer/sparse_input_layer/input_layer/C1_embedding/embedding_weights/Initializer/truncated_normal/shapeConst*
_output_shapes
:*
valueB"'     *
dtype0*\
_classR
PNloc:@input_layer/sparse_input_layer/input_layer/C1_embedding/embedding_weights
О
kinput_layer/sparse_input_layer/input_layer/C1_embedding/embedding_weights/Initializer/truncated_normal/meanConst*
dtype0*
valueB
 *    *\
_classR
PNloc:@input_layer/sparse_input_layer/input_layer/C1_embedding/embedding_weights*
_output_shapes
: 
Р
minput_layer/sparse_input_layer/input_layer/C1_embedding/embedding_weights/Initializer/truncated_normal/stddevConst*\
_classR
PNloc:@input_layer/sparse_input_layer/input_layer/C1_embedding/embedding_weights*
valueB
 *  А>*
dtype0*
_output_shapes
: 
М
vinput_layer/sparse_input_layer/input_layer/C1_embedding/embedding_weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormallinput_layer/sparse_input_layer/input_layer/C1_embedding/embedding_weights/Initializer/truncated_normal/shape*
T0*
_output_shapes
:	РN*\
_classR
PNloc:@input_layer/sparse_input_layer/input_layer/C1_embedding/embedding_weights*
dtype0
а
jinput_layer/sparse_input_layer/input_layer/C1_embedding/embedding_weights/Initializer/truncated_normal/mulMulvinput_layer/sparse_input_layer/input_layer/C1_embedding/embedding_weights/Initializer/truncated_normal/TruncatedNormalminput_layer/sparse_input_layer/input_layer/C1_embedding/embedding_weights/Initializer/truncated_normal/stddev*\
_classR
PNloc:@input_layer/sparse_input_layer/input_layer/C1_embedding/embedding_weights*
_output_shapes
:	РN*
T0
ќ
finput_layer/sparse_input_layer/input_layer/C1_embedding/embedding_weights/Initializer/truncated_normalAddjinput_layer/sparse_input_layer/input_layer/C1_embedding/embedding_weights/Initializer/truncated_normal/mulkinput_layer/sparse_input_layer/input_layer/C1_embedding/embedding_weights/Initializer/truncated_normal/mean*\
_classR
PNloc:@input_layer/sparse_input_layer/input_layer/C1_embedding/embedding_weights*
T0*
_output_shapes
:	РN
щ
Iinput_layer/sparse_input_layer/input_layer/C1_embedding/embedding_weights
VariableV2*
_output_shapes
:	РN*
shape:	РN*
dtype0*\
_classR
PNloc:@input_layer/sparse_input_layer/input_layer/C1_embedding/embedding_weights
Х
Pinput_layer/sparse_input_layer/input_layer/C1_embedding/embedding_weights/AssignAssignIinput_layer/sparse_input_layer/input_layer/C1_embedding/embedding_weightsfinput_layer/sparse_input_layer/input_layer/C1_embedding/embedding_weights/Initializer/truncated_normal*\
_classR
PNloc:@input_layer/sparse_input_layer/input_layer/C1_embedding/embedding_weights*
_output_shapes
:	РN*
T0
≠
Ninput_layer/sparse_input_layer/input_layer/C1_embedding/embedding_weights/readIdentityIinput_layer/sparse_input_layer/input_layer/C1_embedding/embedding_weights*\
_classR
PNloc:@input_layer/sparse_input_layer/input_layer/C1_embedding/embedding_weights*
T0*
_output_shapes
:	РN
Ґ
Xinput_layer/sparse_input_layer/input_layer/C1_embedding/C1_embedding_weights/Slice/beginConst*
_output_shapes
:*
dtype0*
valueB: 
°
Winput_layer/sparse_input_layer/input_layer/C1_embedding/C1_embedding_weights/Slice/sizeConst*
dtype0*
_output_shapes
:*
valueB:
Х
Rinput_layer/sparse_input_layer/input_layer/C1_embedding/C1_embedding_weights/SliceSliceSinput_layer/sparse_input_layer/input_layer/C1_embedding/to_sparse_input/dense_shapeXinput_layer/sparse_input_layer/input_layer/C1_embedding/C1_embedding_weights/Slice/beginWinput_layer/sparse_input_layer/input_layer/C1_embedding/C1_embedding_weights/Slice/size*
T0	*
_output_shapes
:*
Index0
Ь
Rinput_layer/sparse_input_layer/input_layer/C1_embedding/C1_embedding_weights/ConstConst*
dtype0*
_output_shapes
:*
valueB: 
Ґ
Qinput_layer/sparse_input_layer/input_layer/C1_embedding/C1_embedding_weights/ProdProdRinput_layer/sparse_input_layer/input_layer/C1_embedding/C1_embedding_weights/SliceRinput_layer/sparse_input_layer/input_layer/C1_embedding/C1_embedding_weights/Const*
_output_shapes
: *
T0	
Я
]input_layer/sparse_input_layer/input_layer/C1_embedding/C1_embedding_weights/GatherV2/indicesConst*
value	B :*
_output_shapes
: *
dtype0
Ь
Zinput_layer/sparse_input_layer/input_layer/C1_embedding/C1_embedding_weights/GatherV2/axisConst*
value	B : *
dtype0*
_output_shapes
: 
µ
Uinput_layer/sparse_input_layer/input_layer/C1_embedding/C1_embedding_weights/GatherV2GatherV2Sinput_layer/sparse_input_layer/input_layer/C1_embedding/to_sparse_input/dense_shape]input_layer/sparse_input_layer/input_layer/C1_embedding/C1_embedding_weights/GatherV2/indicesZinput_layer/sparse_input_layer/input_layer/C1_embedding/C1_embedding_weights/GatherV2/axis*
Tindices0*
Tparams0	*
Taxis0*
_output_shapes
: 
≥
Sinput_layer/sparse_input_layer/input_layer/C1_embedding/C1_embedding_weights/Cast/xPackQinput_layer/sparse_input_layer/input_layer/C1_embedding/C1_embedding_weights/ProdUinput_layer/sparse_input_layer/input_layer/C1_embedding/C1_embedding_weights/GatherV2*
N*
T0	*
_output_shapes
:
Х
Zinput_layer/sparse_input_layer/input_layer/C1_embedding/C1_embedding_weights/SparseReshapeSparseReshapeOinput_layer/sparse_input_layer/input_layer/C1_embedding/to_sparse_input/indicesSinput_layer/sparse_input_layer/input_layer/C1_embedding/to_sparse_input/dense_shapeSinput_layer/sparse_input_layer/input_layer/C1_embedding/C1_embedding_weights/Cast/x*-
_output_shapes
:€€€€€€€€€:
Ё
cinput_layer/sparse_input_layer/input_layer/C1_embedding/C1_embedding_weights/SparseReshape/IdentityIdentity>input_layer/sparse_input_layer/input_layer/C1_embedding/lookup*#
_output_shapes
:€€€€€€€€€*
T0	
Э
[input_layer/sparse_input_layer/input_layer/C1_embedding/C1_embedding_weights/GreaterEqual/yConst*
_output_shapes
: *
value	B	 R *
dtype0	
ў
Yinput_layer/sparse_input_layer/input_layer/C1_embedding/C1_embedding_weights/GreaterEqualGreaterEqualcinput_layer/sparse_input_layer/input_layer/C1_embedding/C1_embedding_weights/SparseReshape/Identity[input_layer/sparse_input_layer/input_layer/C1_embedding/C1_embedding_weights/GreaterEqual/y*#
_output_shapes
:€€€€€€€€€*
T0	
я
Rinput_layer/sparse_input_layer/input_layer/C1_embedding/C1_embedding_weights/WhereWhereYinput_layer/sparse_input_layer/input_layer/C1_embedding/C1_embedding_weights/GreaterEqual*'
_output_shapes
:€€€€€€€€€
≠
Zinput_layer/sparse_input_layer/input_layer/C1_embedding/C1_embedding_weights/Reshape/shapeConst*
valueB:
€€€€€€€€€*
_output_shapes
:*
dtype0
љ
Tinput_layer/sparse_input_layer/input_layer/C1_embedding/C1_embedding_weights/ReshapeReshapeRinput_layer/sparse_input_layer/input_layer/C1_embedding/C1_embedding_weights/WhereZinput_layer/sparse_input_layer/input_layer/C1_embedding/C1_embedding_weights/Reshape/shape*#
_output_shapes
:€€€€€€€€€*
T0	
Ю
\input_layer/sparse_input_layer/input_layer/C1_embedding/C1_embedding_weights/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
»
Winput_layer/sparse_input_layer/input_layer/C1_embedding/C1_embedding_weights/GatherV2_1GatherV2Zinput_layer/sparse_input_layer/input_layer/C1_embedding/C1_embedding_weights/SparseReshapeTinput_layer/sparse_input_layer/input_layer/C1_embedding/C1_embedding_weights/Reshape\input_layer/sparse_input_layer/input_layer/C1_embedding/C1_embedding_weights/GatherV2_1/axis*'
_output_shapes
:€€€€€€€€€*
Taxis0*
Tindices0	*
Tparams0	
Ю
\input_layer/sparse_input_layer/input_layer/C1_embedding/C1_embedding_weights/GatherV2_2/axisConst*
dtype0*
_output_shapes
: *
value	B : 
Ќ
Winput_layer/sparse_input_layer/input_layer/C1_embedding/C1_embedding_weights/GatherV2_2GatherV2cinput_layer/sparse_input_layer/input_layer/C1_embedding/C1_embedding_weights/SparseReshape/IdentityTinput_layer/sparse_input_layer/input_layer/C1_embedding/C1_embedding_weights/Reshape\input_layer/sparse_input_layer/input_layer/C1_embedding/C1_embedding_weights/GatherV2_2/axis*#
_output_shapes
:€€€€€€€€€*
Tindices0	*
Tparams0	*
Taxis0
д
Uinput_layer/sparse_input_layer/input_layer/C1_embedding/C1_embedding_weights/IdentityIdentity\input_layer/sparse_input_layer/input_layer/C1_embedding/C1_embedding_weights/SparseReshape:1*
T0	*
_output_shapes
:
®
finput_layer/sparse_input_layer/input_layer/C1_embedding/C1_embedding_weights/SparseFillEmptyRows/ConstConst*
_output_shapes
: *
value	B	 R *
dtype0	
џ
tinput_layer/sparse_input_layer/input_layer/C1_embedding/C1_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowsSparseFillEmptyRowsWinput_layer/sparse_input_layer/input_layer/C1_embedding/C1_embedding_weights/GatherV2_1Winput_layer/sparse_input_layer/input_layer/C1_embedding/C1_embedding_weights/GatherV2_2Uinput_layer/sparse_input_layer/input_layer/C1_embedding/C1_embedding_weights/Identityfinput_layer/sparse_input_layer/input_layer/C1_embedding/C1_embedding_weights/SparseFillEmptyRows/Const*
T0	*T
_output_shapesB
@:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€
…
xinput_layer/sparse_input_layer/input_layer/C1_embedding/C1_embedding_weights/embedding_lookup_sparse/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB"        
Ћ
zinput_layer/sparse_input_layer/input_layer/C1_embedding/C1_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       
Ћ
zinput_layer/sparse_input_layer/input_layer/C1_embedding/C1_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB"      
я
rinput_layer/sparse_input_layer/input_layer/C1_embedding/C1_embedding_weights/embedding_lookup_sparse/strided_sliceStridedSlicetinput_layer/sparse_input_layer/input_layer/C1_embedding/C1_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowsxinput_layer/sparse_input_layer/input_layer/C1_embedding/C1_embedding_weights/embedding_lookup_sparse/strided_slice/stackzinput_layer/sparse_input_layer/input_layer/C1_embedding/C1_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1zinput_layer/sparse_input_layer/input_layer/C1_embedding/C1_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2*

begin_mask*
T0	*#
_output_shapes
:€€€€€€€€€*
Index0*
end_mask*
shrink_axis_mask
Ґ
iinput_layer/sparse_input_layer/input_layer/C1_embedding/C1_embedding_weights/embedding_lookup_sparse/CastCastrinput_layer/sparse_input_layer/input_layer/C1_embedding/C1_embedding_weights/embedding_lookup_sparse/strided_slice*#
_output_shapes
:€€€€€€€€€*

SrcT0	*

DstT0
™
kinput_layer/sparse_input_layer/input_layer/C1_embedding/C1_embedding_weights/embedding_lookup_sparse/UniqueUniquevinput_layer/sparse_input_layer/input_layer/C1_embedding/C1_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:1*
T0	*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
Ъ
zinput_layer/sparse_input_layer/input_layer/C1_embedding/C1_embedding_weights/embedding_lookup_sparse/embedding_lookup/axisConst*
_output_shapes
: *
value	B : *
dtype0*\
_classR
PNloc:@input_layer/sparse_input_layer/input_layer/C1_embedding/embedding_weights
н
uinput_layer/sparse_input_layer/input_layer/C1_embedding/C1_embedding_weights/embedding_lookup_sparse/embedding_lookupGatherV2Ninput_layer/sparse_input_layer/input_layer/C1_embedding/embedding_weights/readkinput_layer/sparse_input_layer/input_layer/C1_embedding/C1_embedding_weights/embedding_lookup_sparse/Uniquezinput_layer/sparse_input_layer/input_layer/C1_embedding/C1_embedding_weights/embedding_lookup_sparse/embedding_lookup/axis*
Taxis0*\
_classR
PNloc:@input_layer/sparse_input_layer/input_layer/C1_embedding/embedding_weights*'
_output_shapes
:€€€€€€€€€*
Tparams0*
Tindices0	
≥
~input_layer/sparse_input_layer/input_layer/C1_embedding/C1_embedding_weights/embedding_lookup_sparse/embedding_lookup/IdentityIdentityuinput_layer/sparse_input_layer/input_layer/C1_embedding/C1_embedding_weights/embedding_lookup_sparse/embedding_lookup*'
_output_shapes
:€€€€€€€€€*
T0
Е
dinput_layer/sparse_input_layer/input_layer/C1_embedding/C1_embedding_weights/embedding_lookup_sparseSparseSegmentMean~input_layer/sparse_input_layer/input_layer/C1_embedding/C1_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identityminput_layer/sparse_input_layer/input_layer/C1_embedding/C1_embedding_weights/embedding_lookup_sparse/Unique:1iinput_layer/sparse_input_layer/input_layer/C1_embedding/C1_embedding_weights/embedding_lookup_sparse/Cast*
T0*'
_output_shapes
:€€€€€€€€€
≠
\input_layer/sparse_input_layer/input_layer/C1_embedding/C1_embedding_weights/Reshape_1/shapeConst*
dtype0*
_output_shapes
:*
valueB"€€€€   
й
Vinput_layer/sparse_input_layer/input_layer/C1_embedding/C1_embedding_weights/Reshape_1Reshapevinput_layer/sparse_input_layer/input_layer/C1_embedding/C1_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:2\input_layer/sparse_input_layer/input_layer/C1_embedding/C1_embedding_weights/Reshape_1/shape*
T0
*'
_output_shapes
:€€€€€€€€€
ж
Rinput_layer/sparse_input_layer/input_layer/C1_embedding/C1_embedding_weights/ShapeShapedinput_layer/sparse_input_layer/input_layer/C1_embedding/C1_embedding_weights/embedding_lookup_sparse*
T0*
_output_shapes
:
™
`input_layer/sparse_input_layer/input_layer/C1_embedding/C1_embedding_weights/strided_slice/stackConst*
valueB:*
dtype0*
_output_shapes
:
ђ
binput_layer/sparse_input_layer/input_layer/C1_embedding/C1_embedding_weights/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ђ
binput_layer/sparse_input_layer/input_layer/C1_embedding/C1_embedding_weights/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
Ѓ
Zinput_layer/sparse_input_layer/input_layer/C1_embedding/C1_embedding_weights/strided_sliceStridedSliceRinput_layer/sparse_input_layer/input_layer/C1_embedding/C1_embedding_weights/Shape`input_layer/sparse_input_layer/input_layer/C1_embedding/C1_embedding_weights/strided_slice/stackbinput_layer/sparse_input_layer/input_layer/C1_embedding/C1_embedding_weights/strided_slice/stack_1binput_layer/sparse_input_layer/input_layer/C1_embedding/C1_embedding_weights/strided_slice/stack_2*
shrink_axis_mask*
_output_shapes
: *
T0*
Index0
Ц
Tinput_layer/sparse_input_layer/input_layer/C1_embedding/C1_embedding_weights/stack/0Const*
dtype0*
value	B :*
_output_shapes
: 
Ї
Rinput_layer/sparse_input_layer/input_layer/C1_embedding/C1_embedding_weights/stackPackTinput_layer/sparse_input_layer/input_layer/C1_embedding/C1_embedding_weights/stack/0Zinput_layer/sparse_input_layer/input_layer/C1_embedding/C1_embedding_weights/strided_slice*
T0*
N*
_output_shapes
:
ј
Qinput_layer/sparse_input_layer/input_layer/C1_embedding/C1_embedding_weights/TileTileVinput_layer/sparse_input_layer/input_layer/C1_embedding/C1_embedding_weights/Reshape_1Rinput_layer/sparse_input_layer/input_layer/C1_embedding/C1_embedding_weights/stack*0
_output_shapes
:€€€€€€€€€€€€€€€€€€*
T0

ь
Winput_layer/sparse_input_layer/input_layer/C1_embedding/C1_embedding_weights/zeros_like	ZerosLikedinput_layer/sparse_input_layer/input_layer/C1_embedding/C1_embedding_weights/embedding_lookup_sparse*'
_output_shapes
:€€€€€€€€€*
T0
Ъ
Linput_layer/sparse_input_layer/input_layer/C1_embedding/C1_embedding_weightsSelectQinput_layer/sparse_input_layer/input_layer/C1_embedding/C1_embedding_weights/TileWinput_layer/sparse_input_layer/input_layer/C1_embedding/C1_embedding_weights/zeros_likedinput_layer/sparse_input_layer/input_layer/C1_embedding/C1_embedding_weights/embedding_lookup_sparse*'
_output_shapes
:€€€€€€€€€*
T0
д
Sinput_layer/sparse_input_layer/input_layer/C1_embedding/C1_embedding_weights/Cast_1CastSinput_layer/sparse_input_layer/input_layer/C1_embedding/to_sparse_input/dense_shape*

SrcT0	*
_output_shapes
:*

DstT0
§
Zinput_layer/sparse_input_layer/input_layer/C1_embedding/C1_embedding_weights/Slice_1/beginConst*
valueB: *
_output_shapes
:*
dtype0
£
Yinput_layer/sparse_input_layer/input_layer/C1_embedding/C1_embedding_weights/Slice_1/sizeConst*
_output_shapes
:*
dtype0*
valueB:
Ы
Tinput_layer/sparse_input_layer/input_layer/C1_embedding/C1_embedding_weights/Slice_1SliceSinput_layer/sparse_input_layer/input_layer/C1_embedding/C1_embedding_weights/Cast_1Zinput_layer/sparse_input_layer/input_layer/C1_embedding/C1_embedding_weights/Slice_1/beginYinput_layer/sparse_input_layer/input_layer/C1_embedding/C1_embedding_weights/Slice_1/size*
_output_shapes
:*
Index0*
T0
–
Tinput_layer/sparse_input_layer/input_layer/C1_embedding/C1_embedding_weights/Shape_1ShapeLinput_layer/sparse_input_layer/input_layer/C1_embedding/C1_embedding_weights*
_output_shapes
:*
T0
§
Zinput_layer/sparse_input_layer/input_layer/C1_embedding/C1_embedding_weights/Slice_2/beginConst*
_output_shapes
:*
dtype0*
valueB:
ђ
Yinput_layer/sparse_input_layer/input_layer/C1_embedding/C1_embedding_weights/Slice_2/sizeConst*
dtype0*
valueB:
€€€€€€€€€*
_output_shapes
:
Ь
Tinput_layer/sparse_input_layer/input_layer/C1_embedding/C1_embedding_weights/Slice_2SliceTinput_layer/sparse_input_layer/input_layer/C1_embedding/C1_embedding_weights/Shape_1Zinput_layer/sparse_input_layer/input_layer/C1_embedding/C1_embedding_weights/Slice_2/beginYinput_layer/sparse_input_layer/input_layer/C1_embedding/C1_embedding_weights/Slice_2/size*
T0*
_output_shapes
:*
Index0
Ъ
Xinput_layer/sparse_input_layer/input_layer/C1_embedding/C1_embedding_weights/concat/axisConst*
value	B : *
_output_shapes
: *
dtype0
У
Sinput_layer/sparse_input_layer/input_layer/C1_embedding/C1_embedding_weights/concatConcatV2Tinput_layer/sparse_input_layer/input_layer/C1_embedding/C1_embedding_weights/Slice_1Tinput_layer/sparse_input_layer/input_layer/C1_embedding/C1_embedding_weights/Slice_2Xinput_layer/sparse_input_layer/input_layer/C1_embedding/C1_embedding_weights/concat/axis*
T0*
N*
_output_shapes
:
ґ
Vinput_layer/sparse_input_layer/input_layer/C1_embedding/C1_embedding_weights/Reshape_2ReshapeLinput_layer/sparse_input_layer/input_layer/C1_embedding/C1_embedding_weightsSinput_layer/sparse_input_layer/input_layer/C1_embedding/C1_embedding_weights/concat*
T0*'
_output_shapes
:€€€€€€€€€
√
=input_layer/sparse_input_layer/input_layer/C1_embedding/ShapeShapeVinput_layer/sparse_input_layer/input_layer/C1_embedding/C1_embedding_weights/Reshape_2*
T0*
_output_shapes
:
Х
Kinput_layer/sparse_input_layer/input_layer/C1_embedding/strided_slice/stackConst*
_output_shapes
:*
valueB: *
dtype0
Ч
Minput_layer/sparse_input_layer/input_layer/C1_embedding/strided_slice/stack_1Const*
dtype0*
valueB:*
_output_shapes
:
Ч
Minput_layer/sparse_input_layer/input_layer/C1_embedding/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
≈
Einput_layer/sparse_input_layer/input_layer/C1_embedding/strided_sliceStridedSlice=input_layer/sparse_input_layer/input_layer/C1_embedding/ShapeKinput_layer/sparse_input_layer/input_layer/C1_embedding/strided_slice/stackMinput_layer/sparse_input_layer/input_layer/C1_embedding/strided_slice/stack_1Minput_layer/sparse_input_layer/input_layer/C1_embedding/strided_slice/stack_2*
T0*
Index0*
_output_shapes
: *
shrink_axis_mask
Й
Ginput_layer/sparse_input_layer/input_layer/C1_embedding/Reshape/shape/1Const*
_output_shapes
: *
value	B :*
dtype0
Л
Einput_layer/sparse_input_layer/input_layer/C1_embedding/Reshape/shapePackEinput_layer/sparse_input_layer/input_layer/C1_embedding/strided_sliceGinput_layer/sparse_input_layer/input_layer/C1_embedding/Reshape/shape/1*
T0*
N*
_output_shapes
:
Ы
?input_layer/sparse_input_layer/input_layer/C1_embedding/ReshapeReshapeVinput_layer/sparse_input_layer/input_layer/C1_embedding/C1_embedding_weights/Reshape_2Einput_layer/sparse_input_layer/input_layer/C1_embedding/Reshape/shape*'
_output_shapes
:€€€€€€€€€*
T0
Т
Ginput_layer/sparse_input_layer/input_layer/C20_embedding/ExpandDims/dimConst*
dtype0*
_output_shapes
: *
valueB :
€€€€€€€€€
№
Cinput_layer/sparse_input_layer/input_layer/C20_embedding/ExpandDims
ExpandDimsPlaceholder_32Ginput_layer/sparse_input_layer/input_layer/C20_embedding/ExpandDims/dim*
T0*'
_output_shapes
:€€€€€€€€€
Ш
Winput_layer/sparse_input_layer/input_layer/C20_embedding/to_sparse_input/ignore_value/xConst*
valueB B *
dtype0*
_output_shapes
: 
≠
Qinput_layer/sparse_input_layer/input_layer/C20_embedding/to_sparse_input/NotEqualNotEqualCinput_layer/sparse_input_layer/input_layer/C20_embedding/ExpandDimsWinput_layer/sparse_input_layer/input_layer/C20_embedding/to_sparse_input/ignore_value/x*
T0*'
_output_shapes
:€€€€€€€€€
’
Pinput_layer/sparse_input_layer/input_layer/C20_embedding/to_sparse_input/indicesWhereQinput_layer/sparse_input_layer/input_layer/C20_embedding/to_sparse_input/NotEqual*'
_output_shapes
:€€€€€€€€€
ґ
Oinput_layer/sparse_input_layer/input_layer/C20_embedding/to_sparse_input/valuesGatherNdCinput_layer/sparse_input_layer/input_layer/C20_embedding/ExpandDimsPinput_layer/sparse_input_layer/input_layer/C20_embedding/to_sparse_input/indices*
Tindices0	*#
_output_shapes
:€€€€€€€€€*
Tparams0
„
Tinput_layer/sparse_input_layer/input_layer/C20_embedding/to_sparse_input/dense_shapeShapeCinput_layer/sparse_input_layer/input_layer/C20_embedding/ExpandDims*
out_type0	*
_output_shapes
:*
T0
г
?input_layer/sparse_input_layer/input_layer/C20_embedding/lookupStringToHashBucketFastOinput_layer/sparse_input_layer/input_layer/C20_embedding/to_sparse_input/values*#
_output_shapes
:€€€€€€€€€*
num_bucketsРN
Э
minput_layer/sparse_input_layer/input_layer/C20_embedding/embedding_weights/Initializer/truncated_normal/shapeConst*
valueB"'     *]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C20_embedding/embedding_weights*
dtype0*
_output_shapes
:
Р
linput_layer/sparse_input_layer/input_layer/C20_embedding/embedding_weights/Initializer/truncated_normal/meanConst*
valueB
 *    *
dtype0*]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C20_embedding/embedding_weights*
_output_shapes
: 
Т
ninput_layer/sparse_input_layer/input_layer/C20_embedding/embedding_weights/Initializer/truncated_normal/stddevConst*
dtype0*]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C20_embedding/embedding_weights*
_output_shapes
: *
valueB
 *  А>
П
winput_layer/sparse_input_layer/input_layer/C20_embedding/embedding_weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormalminput_layer/sparse_input_layer/input_layer/C20_embedding/embedding_weights/Initializer/truncated_normal/shape*
dtype0*]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C20_embedding/embedding_weights*
T0*
_output_shapes
:	РN
д
kinput_layer/sparse_input_layer/input_layer/C20_embedding/embedding_weights/Initializer/truncated_normal/mulMulwinput_layer/sparse_input_layer/input_layer/C20_embedding/embedding_weights/Initializer/truncated_normal/TruncatedNormalninput_layer/sparse_input_layer/input_layer/C20_embedding/embedding_weights/Initializer/truncated_normal/stddev*
T0*
_output_shapes
:	РN*]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C20_embedding/embedding_weights
“
ginput_layer/sparse_input_layer/input_layer/C20_embedding/embedding_weights/Initializer/truncated_normalAddkinput_layer/sparse_input_layer/input_layer/C20_embedding/embedding_weights/Initializer/truncated_normal/mullinput_layer/sparse_input_layer/input_layer/C20_embedding/embedding_weights/Initializer/truncated_normal/mean*]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C20_embedding/embedding_weights*
_output_shapes
:	РN*
T0
ы
Jinput_layer/sparse_input_layer/input_layer/C20_embedding/embedding_weights
VariableV2*
_output_shapes
:	РN*
shape:	РN*]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C20_embedding/embedding_weights*
dtype0
Щ
Qinput_layer/sparse_input_layer/input_layer/C20_embedding/embedding_weights/AssignAssignJinput_layer/sparse_input_layer/input_layer/C20_embedding/embedding_weightsginput_layer/sparse_input_layer/input_layer/C20_embedding/embedding_weights/Initializer/truncated_normal*
T0*]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C20_embedding/embedding_weights*
_output_shapes
:	РN
∞
Oinput_layer/sparse_input_layer/input_layer/C20_embedding/embedding_weights/readIdentityJinput_layer/sparse_input_layer/input_layer/C20_embedding/embedding_weights*
T0*
_output_shapes
:	РN*]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C20_embedding/embedding_weights
§
Zinput_layer/sparse_input_layer/input_layer/C20_embedding/C20_embedding_weights/Slice/beginConst*
dtype0*
valueB: *
_output_shapes
:
£
Yinput_layer/sparse_input_layer/input_layer/C20_embedding/C20_embedding_weights/Slice/sizeConst*
_output_shapes
:*
valueB:*
dtype0
Ь
Tinput_layer/sparse_input_layer/input_layer/C20_embedding/C20_embedding_weights/SliceSliceTinput_layer/sparse_input_layer/input_layer/C20_embedding/to_sparse_input/dense_shapeZinput_layer/sparse_input_layer/input_layer/C20_embedding/C20_embedding_weights/Slice/beginYinput_layer/sparse_input_layer/input_layer/C20_embedding/C20_embedding_weights/Slice/size*
_output_shapes
:*
T0	*
Index0
Ю
Tinput_layer/sparse_input_layer/input_layer/C20_embedding/C20_embedding_weights/ConstConst*
_output_shapes
:*
valueB: *
dtype0
®
Sinput_layer/sparse_input_layer/input_layer/C20_embedding/C20_embedding_weights/ProdProdTinput_layer/sparse_input_layer/input_layer/C20_embedding/C20_embedding_weights/SliceTinput_layer/sparse_input_layer/input_layer/C20_embedding/C20_embedding_weights/Const*
T0	*
_output_shapes
: 
°
_input_layer/sparse_input_layer/input_layer/C20_embedding/C20_embedding_weights/GatherV2/indicesConst*
dtype0*
value	B :*
_output_shapes
: 
Ю
\input_layer/sparse_input_layer/input_layer/C20_embedding/C20_embedding_weights/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 
Љ
Winput_layer/sparse_input_layer/input_layer/C20_embedding/C20_embedding_weights/GatherV2GatherV2Tinput_layer/sparse_input_layer/input_layer/C20_embedding/to_sparse_input/dense_shape_input_layer/sparse_input_layer/input_layer/C20_embedding/C20_embedding_weights/GatherV2/indices\input_layer/sparse_input_layer/input_layer/C20_embedding/C20_embedding_weights/GatherV2/axis*
Tparams0	*
_output_shapes
: *
Taxis0*
Tindices0
є
Uinput_layer/sparse_input_layer/input_layer/C20_embedding/C20_embedding_weights/Cast/xPackSinput_layer/sparse_input_layer/input_layer/C20_embedding/C20_embedding_weights/ProdWinput_layer/sparse_input_layer/input_layer/C20_embedding/C20_embedding_weights/GatherV2*
_output_shapes
:*
T0	*
N
Ы
\input_layer/sparse_input_layer/input_layer/C20_embedding/C20_embedding_weights/SparseReshapeSparseReshapePinput_layer/sparse_input_layer/input_layer/C20_embedding/to_sparse_input/indicesTinput_layer/sparse_input_layer/input_layer/C20_embedding/to_sparse_input/dense_shapeUinput_layer/sparse_input_layer/input_layer/C20_embedding/C20_embedding_weights/Cast/x*-
_output_shapes
:€€€€€€€€€:
а
einput_layer/sparse_input_layer/input_layer/C20_embedding/C20_embedding_weights/SparseReshape/IdentityIdentity?input_layer/sparse_input_layer/input_layer/C20_embedding/lookup*#
_output_shapes
:€€€€€€€€€*
T0	
Я
]input_layer/sparse_input_layer/input_layer/C20_embedding/C20_embedding_weights/GreaterEqual/yConst*
value	B	 R *
dtype0	*
_output_shapes
: 
я
[input_layer/sparse_input_layer/input_layer/C20_embedding/C20_embedding_weights/GreaterEqualGreaterEqualeinput_layer/sparse_input_layer/input_layer/C20_embedding/C20_embedding_weights/SparseReshape/Identity]input_layer/sparse_input_layer/input_layer/C20_embedding/C20_embedding_weights/GreaterEqual/y*
T0	*#
_output_shapes
:€€€€€€€€€
г
Tinput_layer/sparse_input_layer/input_layer/C20_embedding/C20_embedding_weights/WhereWhere[input_layer/sparse_input_layer/input_layer/C20_embedding/C20_embedding_weights/GreaterEqual*'
_output_shapes
:€€€€€€€€€
ѓ
\input_layer/sparse_input_layer/input_layer/C20_embedding/C20_embedding_weights/Reshape/shapeConst*
dtype0*
_output_shapes
:*
valueB:
€€€€€€€€€
√
Vinput_layer/sparse_input_layer/input_layer/C20_embedding/C20_embedding_weights/ReshapeReshapeTinput_layer/sparse_input_layer/input_layer/C20_embedding/C20_embedding_weights/Where\input_layer/sparse_input_layer/input_layer/C20_embedding/C20_embedding_weights/Reshape/shape*#
_output_shapes
:€€€€€€€€€*
T0	
†
^input_layer/sparse_input_layer/input_layer/C20_embedding/C20_embedding_weights/GatherV2_1/axisConst*
value	B : *
_output_shapes
: *
dtype0
–
Yinput_layer/sparse_input_layer/input_layer/C20_embedding/C20_embedding_weights/GatherV2_1GatherV2\input_layer/sparse_input_layer/input_layer/C20_embedding/C20_embedding_weights/SparseReshapeVinput_layer/sparse_input_layer/input_layer/C20_embedding/C20_embedding_weights/Reshape^input_layer/sparse_input_layer/input_layer/C20_embedding/C20_embedding_weights/GatherV2_1/axis*
Taxis0*
Tindices0	*'
_output_shapes
:€€€€€€€€€*
Tparams0	
†
^input_layer/sparse_input_layer/input_layer/C20_embedding/C20_embedding_weights/GatherV2_2/axisConst*
value	B : *
dtype0*
_output_shapes
: 
’
Yinput_layer/sparse_input_layer/input_layer/C20_embedding/C20_embedding_weights/GatherV2_2GatherV2einput_layer/sparse_input_layer/input_layer/C20_embedding/C20_embedding_weights/SparseReshape/IdentityVinput_layer/sparse_input_layer/input_layer/C20_embedding/C20_embedding_weights/Reshape^input_layer/sparse_input_layer/input_layer/C20_embedding/C20_embedding_weights/GatherV2_2/axis*
Tindices0	*#
_output_shapes
:€€€€€€€€€*
Tparams0	*
Taxis0
и
Winput_layer/sparse_input_layer/input_layer/C20_embedding/C20_embedding_weights/IdentityIdentity^input_layer/sparse_input_layer/input_layer/C20_embedding/C20_embedding_weights/SparseReshape:1*
T0	*
_output_shapes
:
™
hinput_layer/sparse_input_layer/input_layer/C20_embedding/C20_embedding_weights/SparseFillEmptyRows/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R 
е
vinput_layer/sparse_input_layer/input_layer/C20_embedding/C20_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowsSparseFillEmptyRowsYinput_layer/sparse_input_layer/input_layer/C20_embedding/C20_embedding_weights/GatherV2_1Yinput_layer/sparse_input_layer/input_layer/C20_embedding/C20_embedding_weights/GatherV2_2Winput_layer/sparse_input_layer/input_layer/C20_embedding/C20_embedding_weights/Identityhinput_layer/sparse_input_layer/input_layer/C20_embedding/C20_embedding_weights/SparseFillEmptyRows/Const*T
_output_shapesB
@:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€*
T0	
Ћ
zinput_layer/sparse_input_layer/input_layer/C20_embedding/C20_embedding_weights/embedding_lookup_sparse/strided_slice/stackConst*
_output_shapes
:*
valueB"        *
dtype0
Ќ
|input_layer/sparse_input_layer/input_layer/C20_embedding/C20_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1Const*
dtype0*
valueB"       *
_output_shapes
:
Ќ
|input_layer/sparse_input_layer/input_layer/C20_embedding/C20_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2Const*
_output_shapes
:*
valueB"      *
dtype0
й
tinput_layer/sparse_input_layer/input_layer/C20_embedding/C20_embedding_weights/embedding_lookup_sparse/strided_sliceStridedSlicevinput_layer/sparse_input_layer/input_layer/C20_embedding/C20_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowszinput_layer/sparse_input_layer/input_layer/C20_embedding/C20_embedding_weights/embedding_lookup_sparse/strided_slice/stack|input_layer/sparse_input_layer/input_layer/C20_embedding/C20_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1|input_layer/sparse_input_layer/input_layer/C20_embedding/C20_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2*

begin_mask*
T0	*
end_mask*#
_output_shapes
:€€€€€€€€€*
shrink_axis_mask*
Index0
¶
kinput_layer/sparse_input_layer/input_layer/C20_embedding/C20_embedding_weights/embedding_lookup_sparse/CastCasttinput_layer/sparse_input_layer/input_layer/C20_embedding/C20_embedding_weights/embedding_lookup_sparse/strided_slice*

DstT0*#
_output_shapes
:€€€€€€€€€*

SrcT0	
Ѓ
minput_layer/sparse_input_layer/input_layer/C20_embedding/C20_embedding_weights/embedding_lookup_sparse/UniqueUniquexinput_layer/sparse_input_layer/input_layer/C20_embedding/C20_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€*
T0	
Э
|input_layer/sparse_input_layer/input_layer/C20_embedding/C20_embedding_weights/embedding_lookup_sparse/embedding_lookup/axisConst*
value	B : *
_output_shapes
: *
dtype0*]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C20_embedding/embedding_weights
х
winput_layer/sparse_input_layer/input_layer/C20_embedding/C20_embedding_weights/embedding_lookup_sparse/embedding_lookupGatherV2Oinput_layer/sparse_input_layer/input_layer/C20_embedding/embedding_weights/readminput_layer/sparse_input_layer/input_layer/C20_embedding/C20_embedding_weights/embedding_lookup_sparse/Unique|input_layer/sparse_input_layer/input_layer/C20_embedding/C20_embedding_weights/embedding_lookup_sparse/embedding_lookup/axis*
Tindices0	*'
_output_shapes
:€€€€€€€€€*
Taxis0*]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C20_embedding/embedding_weights*
Tparams0
Є
Аinput_layer/sparse_input_layer/input_layer/C20_embedding/C20_embedding_weights/embedding_lookup_sparse/embedding_lookup/IdentityIdentitywinput_layer/sparse_input_layer/input_layer/C20_embedding/C20_embedding_weights/embedding_lookup_sparse/embedding_lookup*'
_output_shapes
:€€€€€€€€€*
T0
О
finput_layer/sparse_input_layer/input_layer/C20_embedding/C20_embedding_weights/embedding_lookup_sparseSparseSegmentMeanАinput_layer/sparse_input_layer/input_layer/C20_embedding/C20_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identityoinput_layer/sparse_input_layer/input_layer/C20_embedding/C20_embedding_weights/embedding_lookup_sparse/Unique:1kinput_layer/sparse_input_layer/input_layer/C20_embedding/C20_embedding_weights/embedding_lookup_sparse/Cast*
T0*'
_output_shapes
:€€€€€€€€€
ѓ
^input_layer/sparse_input_layer/input_layer/C20_embedding/C20_embedding_weights/Reshape_1/shapeConst*
dtype0*
valueB"€€€€   *
_output_shapes
:
п
Xinput_layer/sparse_input_layer/input_layer/C20_embedding/C20_embedding_weights/Reshape_1Reshapexinput_layer/sparse_input_layer/input_layer/C20_embedding/C20_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:2^input_layer/sparse_input_layer/input_layer/C20_embedding/C20_embedding_weights/Reshape_1/shape*
T0
*'
_output_shapes
:€€€€€€€€€
к
Tinput_layer/sparse_input_layer/input_layer/C20_embedding/C20_embedding_weights/ShapeShapefinput_layer/sparse_input_layer/input_layer/C20_embedding/C20_embedding_weights/embedding_lookup_sparse*
T0*
_output_shapes
:
ђ
binput_layer/sparse_input_layer/input_layer/C20_embedding/C20_embedding_weights/strided_slice/stackConst*
valueB:*
dtype0*
_output_shapes
:
Ѓ
dinput_layer/sparse_input_layer/input_layer/C20_embedding/C20_embedding_weights/strided_slice/stack_1Const*
dtype0*
valueB:*
_output_shapes
:
Ѓ
dinput_layer/sparse_input_layer/input_layer/C20_embedding/C20_embedding_weights/strided_slice/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
Є
\input_layer/sparse_input_layer/input_layer/C20_embedding/C20_embedding_weights/strided_sliceStridedSliceTinput_layer/sparse_input_layer/input_layer/C20_embedding/C20_embedding_weights/Shapebinput_layer/sparse_input_layer/input_layer/C20_embedding/C20_embedding_weights/strided_slice/stackdinput_layer/sparse_input_layer/input_layer/C20_embedding/C20_embedding_weights/strided_slice/stack_1dinput_layer/sparse_input_layer/input_layer/C20_embedding/C20_embedding_weights/strided_slice/stack_2*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: 
Ш
Vinput_layer/sparse_input_layer/input_layer/C20_embedding/C20_embedding_weights/stack/0Const*
dtype0*
value	B :*
_output_shapes
: 
ј
Tinput_layer/sparse_input_layer/input_layer/C20_embedding/C20_embedding_weights/stackPackVinput_layer/sparse_input_layer/input_layer/C20_embedding/C20_embedding_weights/stack/0\input_layer/sparse_input_layer/input_layer/C20_embedding/C20_embedding_weights/strided_slice*
N*
T0*
_output_shapes
:
∆
Sinput_layer/sparse_input_layer/input_layer/C20_embedding/C20_embedding_weights/TileTileXinput_layer/sparse_input_layer/input_layer/C20_embedding/C20_embedding_weights/Reshape_1Tinput_layer/sparse_input_layer/input_layer/C20_embedding/C20_embedding_weights/stack*
T0
*0
_output_shapes
:€€€€€€€€€€€€€€€€€€
А
Yinput_layer/sparse_input_layer/input_layer/C20_embedding/C20_embedding_weights/zeros_like	ZerosLikefinput_layer/sparse_input_layer/input_layer/C20_embedding/C20_embedding_weights/embedding_lookup_sparse*
T0*'
_output_shapes
:€€€€€€€€€
Ґ
Ninput_layer/sparse_input_layer/input_layer/C20_embedding/C20_embedding_weightsSelectSinput_layer/sparse_input_layer/input_layer/C20_embedding/C20_embedding_weights/TileYinput_layer/sparse_input_layer/input_layer/C20_embedding/C20_embedding_weights/zeros_likefinput_layer/sparse_input_layer/input_layer/C20_embedding/C20_embedding_weights/embedding_lookup_sparse*'
_output_shapes
:€€€€€€€€€*
T0
з
Uinput_layer/sparse_input_layer/input_layer/C20_embedding/C20_embedding_weights/Cast_1CastTinput_layer/sparse_input_layer/input_layer/C20_embedding/to_sparse_input/dense_shape*

DstT0*
_output_shapes
:*

SrcT0	
¶
\input_layer/sparse_input_layer/input_layer/C20_embedding/C20_embedding_weights/Slice_1/beginConst*
valueB: *
_output_shapes
:*
dtype0
•
[input_layer/sparse_input_layer/input_layer/C20_embedding/C20_embedding_weights/Slice_1/sizeConst*
dtype0*
_output_shapes
:*
valueB:
£
Vinput_layer/sparse_input_layer/input_layer/C20_embedding/C20_embedding_weights/Slice_1SliceUinput_layer/sparse_input_layer/input_layer/C20_embedding/C20_embedding_weights/Cast_1\input_layer/sparse_input_layer/input_layer/C20_embedding/C20_embedding_weights/Slice_1/begin[input_layer/sparse_input_layer/input_layer/C20_embedding/C20_embedding_weights/Slice_1/size*
Index0*
_output_shapes
:*
T0
‘
Vinput_layer/sparse_input_layer/input_layer/C20_embedding/C20_embedding_weights/Shape_1ShapeNinput_layer/sparse_input_layer/input_layer/C20_embedding/C20_embedding_weights*
T0*
_output_shapes
:
¶
\input_layer/sparse_input_layer/input_layer/C20_embedding/C20_embedding_weights/Slice_2/beginConst*
_output_shapes
:*
dtype0*
valueB:
Ѓ
[input_layer/sparse_input_layer/input_layer/C20_embedding/C20_embedding_weights/Slice_2/sizeConst*
valueB:
€€€€€€€€€*
_output_shapes
:*
dtype0
§
Vinput_layer/sparse_input_layer/input_layer/C20_embedding/C20_embedding_weights/Slice_2SliceVinput_layer/sparse_input_layer/input_layer/C20_embedding/C20_embedding_weights/Shape_1\input_layer/sparse_input_layer/input_layer/C20_embedding/C20_embedding_weights/Slice_2/begin[input_layer/sparse_input_layer/input_layer/C20_embedding/C20_embedding_weights/Slice_2/size*
Index0*
_output_shapes
:*
T0
Ь
Zinput_layer/sparse_input_layer/input_layer/C20_embedding/C20_embedding_weights/concat/axisConst*
value	B : *
_output_shapes
: *
dtype0
Ы
Uinput_layer/sparse_input_layer/input_layer/C20_embedding/C20_embedding_weights/concatConcatV2Vinput_layer/sparse_input_layer/input_layer/C20_embedding/C20_embedding_weights/Slice_1Vinput_layer/sparse_input_layer/input_layer/C20_embedding/C20_embedding_weights/Slice_2Zinput_layer/sparse_input_layer/input_layer/C20_embedding/C20_embedding_weights/concat/axis*
T0*
_output_shapes
:*
N
Љ
Xinput_layer/sparse_input_layer/input_layer/C20_embedding/C20_embedding_weights/Reshape_2ReshapeNinput_layer/sparse_input_layer/input_layer/C20_embedding/C20_embedding_weightsUinput_layer/sparse_input_layer/input_layer/C20_embedding/C20_embedding_weights/concat*'
_output_shapes
:€€€€€€€€€*
T0
∆
>input_layer/sparse_input_layer/input_layer/C20_embedding/ShapeShapeXinput_layer/sparse_input_layer/input_layer/C20_embedding/C20_embedding_weights/Reshape_2*
T0*
_output_shapes
:
Ц
Linput_layer/sparse_input_layer/input_layer/C20_embedding/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Ш
Ninput_layer/sparse_input_layer/input_layer/C20_embedding/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Ш
Ninput_layer/sparse_input_layer/input_layer/C20_embedding/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
 
Finput_layer/sparse_input_layer/input_layer/C20_embedding/strided_sliceStridedSlice>input_layer/sparse_input_layer/input_layer/C20_embedding/ShapeLinput_layer/sparse_input_layer/input_layer/C20_embedding/strided_slice/stackNinput_layer/sparse_input_layer/input_layer/C20_embedding/strided_slice/stack_1Ninput_layer/sparse_input_layer/input_layer/C20_embedding/strided_slice/stack_2*
T0*
shrink_axis_mask*
Index0*
_output_shapes
: 
К
Hinput_layer/sparse_input_layer/input_layer/C20_embedding/Reshape/shape/1Const*
value	B :*
dtype0*
_output_shapes
: 
О
Finput_layer/sparse_input_layer/input_layer/C20_embedding/Reshape/shapePackFinput_layer/sparse_input_layer/input_layer/C20_embedding/strided_sliceHinput_layer/sparse_input_layer/input_layer/C20_embedding/Reshape/shape/1*
N*
_output_shapes
:*
T0
Я
@input_layer/sparse_input_layer/input_layer/C20_embedding/ReshapeReshapeXinput_layer/sparse_input_layer/input_layer/C20_embedding/C20_embedding_weights/Reshape_2Finput_layer/sparse_input_layer/input_layer/C20_embedding/Reshape/shape*'
_output_shapes
:€€€€€€€€€*
T0
Т
Ginput_layer/sparse_input_layer/input_layer/C21_embedding/ExpandDims/dimConst*
valueB :
€€€€€€€€€*
_output_shapes
: *
dtype0
№
Cinput_layer/sparse_input_layer/input_layer/C21_embedding/ExpandDims
ExpandDimsPlaceholder_33Ginput_layer/sparse_input_layer/input_layer/C21_embedding/ExpandDims/dim*
T0*'
_output_shapes
:€€€€€€€€€
Ш
Winput_layer/sparse_input_layer/input_layer/C21_embedding/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
valueB B *
dtype0
≠
Qinput_layer/sparse_input_layer/input_layer/C21_embedding/to_sparse_input/NotEqualNotEqualCinput_layer/sparse_input_layer/input_layer/C21_embedding/ExpandDimsWinput_layer/sparse_input_layer/input_layer/C21_embedding/to_sparse_input/ignore_value/x*'
_output_shapes
:€€€€€€€€€*
T0
’
Pinput_layer/sparse_input_layer/input_layer/C21_embedding/to_sparse_input/indicesWhereQinput_layer/sparse_input_layer/input_layer/C21_embedding/to_sparse_input/NotEqual*'
_output_shapes
:€€€€€€€€€
ґ
Oinput_layer/sparse_input_layer/input_layer/C21_embedding/to_sparse_input/valuesGatherNdCinput_layer/sparse_input_layer/input_layer/C21_embedding/ExpandDimsPinput_layer/sparse_input_layer/input_layer/C21_embedding/to_sparse_input/indices*
Tindices0	*
Tparams0*#
_output_shapes
:€€€€€€€€€
„
Tinput_layer/sparse_input_layer/input_layer/C21_embedding/to_sparse_input/dense_shapeShapeCinput_layer/sparse_input_layer/input_layer/C21_embedding/ExpandDims*
out_type0	*
_output_shapes
:*
T0
г
?input_layer/sparse_input_layer/input_layer/C21_embedding/lookupStringToHashBucketFastOinput_layer/sparse_input_layer/input_layer/C21_embedding/to_sparse_input/values*
num_bucketsРN*#
_output_shapes
:€€€€€€€€€
Э
minput_layer/sparse_input_layer/input_layer/C21_embedding/embedding_weights/Initializer/truncated_normal/shapeConst*
valueB"'     *]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C21_embedding/embedding_weights*
dtype0*
_output_shapes
:
Р
linput_layer/sparse_input_layer/input_layer/C21_embedding/embedding_weights/Initializer/truncated_normal/meanConst*]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C21_embedding/embedding_weights*
valueB
 *    *
_output_shapes
: *
dtype0
Т
ninput_layer/sparse_input_layer/input_layer/C21_embedding/embedding_weights/Initializer/truncated_normal/stddevConst*]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C21_embedding/embedding_weights*
_output_shapes
: *
valueB
 *  А>*
dtype0
П
winput_layer/sparse_input_layer/input_layer/C21_embedding/embedding_weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormalminput_layer/sparse_input_layer/input_layer/C21_embedding/embedding_weights/Initializer/truncated_normal/shape*
_output_shapes
:	РN*
dtype0*]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C21_embedding/embedding_weights*
T0
д
kinput_layer/sparse_input_layer/input_layer/C21_embedding/embedding_weights/Initializer/truncated_normal/mulMulwinput_layer/sparse_input_layer/input_layer/C21_embedding/embedding_weights/Initializer/truncated_normal/TruncatedNormalninput_layer/sparse_input_layer/input_layer/C21_embedding/embedding_weights/Initializer/truncated_normal/stddev*]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C21_embedding/embedding_weights*
_output_shapes
:	РN*
T0
“
ginput_layer/sparse_input_layer/input_layer/C21_embedding/embedding_weights/Initializer/truncated_normalAddkinput_layer/sparse_input_layer/input_layer/C21_embedding/embedding_weights/Initializer/truncated_normal/mullinput_layer/sparse_input_layer/input_layer/C21_embedding/embedding_weights/Initializer/truncated_normal/mean*]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C21_embedding/embedding_weights*
T0*
_output_shapes
:	РN
ы
Jinput_layer/sparse_input_layer/input_layer/C21_embedding/embedding_weights
VariableV2*
dtype0*
shape:	РN*
_output_shapes
:	РN*]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C21_embedding/embedding_weights
Щ
Qinput_layer/sparse_input_layer/input_layer/C21_embedding/embedding_weights/AssignAssignJinput_layer/sparse_input_layer/input_layer/C21_embedding/embedding_weightsginput_layer/sparse_input_layer/input_layer/C21_embedding/embedding_weights/Initializer/truncated_normal*]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C21_embedding/embedding_weights*
T0*
_output_shapes
:	РN
∞
Oinput_layer/sparse_input_layer/input_layer/C21_embedding/embedding_weights/readIdentityJinput_layer/sparse_input_layer/input_layer/C21_embedding/embedding_weights*]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C21_embedding/embedding_weights*
_output_shapes
:	РN*
T0
§
Zinput_layer/sparse_input_layer/input_layer/C21_embedding/C21_embedding_weights/Slice/beginConst*
valueB: *
_output_shapes
:*
dtype0
£
Yinput_layer/sparse_input_layer/input_layer/C21_embedding/C21_embedding_weights/Slice/sizeConst*
_output_shapes
:*
dtype0*
valueB:
Ь
Tinput_layer/sparse_input_layer/input_layer/C21_embedding/C21_embedding_weights/SliceSliceTinput_layer/sparse_input_layer/input_layer/C21_embedding/to_sparse_input/dense_shapeZinput_layer/sparse_input_layer/input_layer/C21_embedding/C21_embedding_weights/Slice/beginYinput_layer/sparse_input_layer/input_layer/C21_embedding/C21_embedding_weights/Slice/size*
T0	*
_output_shapes
:*
Index0
Ю
Tinput_layer/sparse_input_layer/input_layer/C21_embedding/C21_embedding_weights/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
®
Sinput_layer/sparse_input_layer/input_layer/C21_embedding/C21_embedding_weights/ProdProdTinput_layer/sparse_input_layer/input_layer/C21_embedding/C21_embedding_weights/SliceTinput_layer/sparse_input_layer/input_layer/C21_embedding/C21_embedding_weights/Const*
_output_shapes
: *
T0	
°
_input_layer/sparse_input_layer/input_layer/C21_embedding/C21_embedding_weights/GatherV2/indicesConst*
dtype0*
_output_shapes
: *
value	B :
Ю
\input_layer/sparse_input_layer/input_layer/C21_embedding/C21_embedding_weights/GatherV2/axisConst*
dtype0*
_output_shapes
: *
value	B : 
Љ
Winput_layer/sparse_input_layer/input_layer/C21_embedding/C21_embedding_weights/GatherV2GatherV2Tinput_layer/sparse_input_layer/input_layer/C21_embedding/to_sparse_input/dense_shape_input_layer/sparse_input_layer/input_layer/C21_embedding/C21_embedding_weights/GatherV2/indices\input_layer/sparse_input_layer/input_layer/C21_embedding/C21_embedding_weights/GatherV2/axis*
Taxis0*
Tindices0*
Tparams0	*
_output_shapes
: 
є
Uinput_layer/sparse_input_layer/input_layer/C21_embedding/C21_embedding_weights/Cast/xPackSinput_layer/sparse_input_layer/input_layer/C21_embedding/C21_embedding_weights/ProdWinput_layer/sparse_input_layer/input_layer/C21_embedding/C21_embedding_weights/GatherV2*
T0	*
N*
_output_shapes
:
Ы
\input_layer/sparse_input_layer/input_layer/C21_embedding/C21_embedding_weights/SparseReshapeSparseReshapePinput_layer/sparse_input_layer/input_layer/C21_embedding/to_sparse_input/indicesTinput_layer/sparse_input_layer/input_layer/C21_embedding/to_sparse_input/dense_shapeUinput_layer/sparse_input_layer/input_layer/C21_embedding/C21_embedding_weights/Cast/x*-
_output_shapes
:€€€€€€€€€:
а
einput_layer/sparse_input_layer/input_layer/C21_embedding/C21_embedding_weights/SparseReshape/IdentityIdentity?input_layer/sparse_input_layer/input_layer/C21_embedding/lookup*
T0	*#
_output_shapes
:€€€€€€€€€
Я
]input_layer/sparse_input_layer/input_layer/C21_embedding/C21_embedding_weights/GreaterEqual/yConst*
_output_shapes
: *
value	B	 R *
dtype0	
я
[input_layer/sparse_input_layer/input_layer/C21_embedding/C21_embedding_weights/GreaterEqualGreaterEqualeinput_layer/sparse_input_layer/input_layer/C21_embedding/C21_embedding_weights/SparseReshape/Identity]input_layer/sparse_input_layer/input_layer/C21_embedding/C21_embedding_weights/GreaterEqual/y*
T0	*#
_output_shapes
:€€€€€€€€€
г
Tinput_layer/sparse_input_layer/input_layer/C21_embedding/C21_embedding_weights/WhereWhere[input_layer/sparse_input_layer/input_layer/C21_embedding/C21_embedding_weights/GreaterEqual*'
_output_shapes
:€€€€€€€€€
ѓ
\input_layer/sparse_input_layer/input_layer/C21_embedding/C21_embedding_weights/Reshape/shapeConst*
dtype0*
valueB:
€€€€€€€€€*
_output_shapes
:
√
Vinput_layer/sparse_input_layer/input_layer/C21_embedding/C21_embedding_weights/ReshapeReshapeTinput_layer/sparse_input_layer/input_layer/C21_embedding/C21_embedding_weights/Where\input_layer/sparse_input_layer/input_layer/C21_embedding/C21_embedding_weights/Reshape/shape*
T0	*#
_output_shapes
:€€€€€€€€€
†
^input_layer/sparse_input_layer/input_layer/C21_embedding/C21_embedding_weights/GatherV2_1/axisConst*
dtype0*
value	B : *
_output_shapes
: 
–
Yinput_layer/sparse_input_layer/input_layer/C21_embedding/C21_embedding_weights/GatherV2_1GatherV2\input_layer/sparse_input_layer/input_layer/C21_embedding/C21_embedding_weights/SparseReshapeVinput_layer/sparse_input_layer/input_layer/C21_embedding/C21_embedding_weights/Reshape^input_layer/sparse_input_layer/input_layer/C21_embedding/C21_embedding_weights/GatherV2_1/axis*
Tindices0	*'
_output_shapes
:€€€€€€€€€*
Taxis0*
Tparams0	
†
^input_layer/sparse_input_layer/input_layer/C21_embedding/C21_embedding_weights/GatherV2_2/axisConst*
_output_shapes
: *
value	B : *
dtype0
’
Yinput_layer/sparse_input_layer/input_layer/C21_embedding/C21_embedding_weights/GatherV2_2GatherV2einput_layer/sparse_input_layer/input_layer/C21_embedding/C21_embedding_weights/SparseReshape/IdentityVinput_layer/sparse_input_layer/input_layer/C21_embedding/C21_embedding_weights/Reshape^input_layer/sparse_input_layer/input_layer/C21_embedding/C21_embedding_weights/GatherV2_2/axis*
Taxis0*#
_output_shapes
:€€€€€€€€€*
Tindices0	*
Tparams0	
и
Winput_layer/sparse_input_layer/input_layer/C21_embedding/C21_embedding_weights/IdentityIdentity^input_layer/sparse_input_layer/input_layer/C21_embedding/C21_embedding_weights/SparseReshape:1*
T0	*
_output_shapes
:
™
hinput_layer/sparse_input_layer/input_layer/C21_embedding/C21_embedding_weights/SparseFillEmptyRows/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R 
е
vinput_layer/sparse_input_layer/input_layer/C21_embedding/C21_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowsSparseFillEmptyRowsYinput_layer/sparse_input_layer/input_layer/C21_embedding/C21_embedding_weights/GatherV2_1Yinput_layer/sparse_input_layer/input_layer/C21_embedding/C21_embedding_weights/GatherV2_2Winput_layer/sparse_input_layer/input_layer/C21_embedding/C21_embedding_weights/Identityhinput_layer/sparse_input_layer/input_layer/C21_embedding/C21_embedding_weights/SparseFillEmptyRows/Const*
T0	*T
_output_shapesB
@:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€
Ћ
zinput_layer/sparse_input_layer/input_layer/C21_embedding/C21_embedding_weights/embedding_lookup_sparse/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        
Ќ
|input_layer/sparse_input_layer/input_layer/C21_embedding/C21_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1Const*
dtype0*
valueB"       *
_output_shapes
:
Ќ
|input_layer/sparse_input_layer/input_layer/C21_embedding/C21_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2Const*
_output_shapes
:*
valueB"      *
dtype0
й
tinput_layer/sparse_input_layer/input_layer/C21_embedding/C21_embedding_weights/embedding_lookup_sparse/strided_sliceStridedSlicevinput_layer/sparse_input_layer/input_layer/C21_embedding/C21_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowszinput_layer/sparse_input_layer/input_layer/C21_embedding/C21_embedding_weights/embedding_lookup_sparse/strided_slice/stack|input_layer/sparse_input_layer/input_layer/C21_embedding/C21_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1|input_layer/sparse_input_layer/input_layer/C21_embedding/C21_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2*
Index0*
shrink_axis_mask*
end_mask*

begin_mask*
T0	*#
_output_shapes
:€€€€€€€€€
¶
kinput_layer/sparse_input_layer/input_layer/C21_embedding/C21_embedding_weights/embedding_lookup_sparse/CastCasttinput_layer/sparse_input_layer/input_layer/C21_embedding/C21_embedding_weights/embedding_lookup_sparse/strided_slice*

DstT0*#
_output_shapes
:€€€€€€€€€*

SrcT0	
Ѓ
minput_layer/sparse_input_layer/input_layer/C21_embedding/C21_embedding_weights/embedding_lookup_sparse/UniqueUniquexinput_layer/sparse_input_layer/input_layer/C21_embedding/C21_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:1*
T0	*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
Э
|input_layer/sparse_input_layer/input_layer/C21_embedding/C21_embedding_weights/embedding_lookup_sparse/embedding_lookup/axisConst*
_output_shapes
: *
dtype0*]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C21_embedding/embedding_weights*
value	B : 
х
winput_layer/sparse_input_layer/input_layer/C21_embedding/C21_embedding_weights/embedding_lookup_sparse/embedding_lookupGatherV2Oinput_layer/sparse_input_layer/input_layer/C21_embedding/embedding_weights/readminput_layer/sparse_input_layer/input_layer/C21_embedding/C21_embedding_weights/embedding_lookup_sparse/Unique|input_layer/sparse_input_layer/input_layer/C21_embedding/C21_embedding_weights/embedding_lookup_sparse/embedding_lookup/axis*
Tparams0*'
_output_shapes
:€€€€€€€€€*
Tindices0	*
Taxis0*]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C21_embedding/embedding_weights
Є
Аinput_layer/sparse_input_layer/input_layer/C21_embedding/C21_embedding_weights/embedding_lookup_sparse/embedding_lookup/IdentityIdentitywinput_layer/sparse_input_layer/input_layer/C21_embedding/C21_embedding_weights/embedding_lookup_sparse/embedding_lookup*'
_output_shapes
:€€€€€€€€€*
T0
О
finput_layer/sparse_input_layer/input_layer/C21_embedding/C21_embedding_weights/embedding_lookup_sparseSparseSegmentMeanАinput_layer/sparse_input_layer/input_layer/C21_embedding/C21_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identityoinput_layer/sparse_input_layer/input_layer/C21_embedding/C21_embedding_weights/embedding_lookup_sparse/Unique:1kinput_layer/sparse_input_layer/input_layer/C21_embedding/C21_embedding_weights/embedding_lookup_sparse/Cast*'
_output_shapes
:€€€€€€€€€*
T0
ѓ
^input_layer/sparse_input_layer/input_layer/C21_embedding/C21_embedding_weights/Reshape_1/shapeConst*
valueB"€€€€   *
_output_shapes
:*
dtype0
п
Xinput_layer/sparse_input_layer/input_layer/C21_embedding/C21_embedding_weights/Reshape_1Reshapexinput_layer/sparse_input_layer/input_layer/C21_embedding/C21_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:2^input_layer/sparse_input_layer/input_layer/C21_embedding/C21_embedding_weights/Reshape_1/shape*
T0
*'
_output_shapes
:€€€€€€€€€
к
Tinput_layer/sparse_input_layer/input_layer/C21_embedding/C21_embedding_weights/ShapeShapefinput_layer/sparse_input_layer/input_layer/C21_embedding/C21_embedding_weights/embedding_lookup_sparse*
_output_shapes
:*
T0
ђ
binput_layer/sparse_input_layer/input_layer/C21_embedding/C21_embedding_weights/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB:
Ѓ
dinput_layer/sparse_input_layer/input_layer/C21_embedding/C21_embedding_weights/strided_slice/stack_1Const*
_output_shapes
:*
valueB:*
dtype0
Ѓ
dinput_layer/sparse_input_layer/input_layer/C21_embedding/C21_embedding_weights/strided_slice/stack_2Const*
_output_shapes
:*
valueB:*
dtype0
Є
\input_layer/sparse_input_layer/input_layer/C21_embedding/C21_embedding_weights/strided_sliceStridedSliceTinput_layer/sparse_input_layer/input_layer/C21_embedding/C21_embedding_weights/Shapebinput_layer/sparse_input_layer/input_layer/C21_embedding/C21_embedding_weights/strided_slice/stackdinput_layer/sparse_input_layer/input_layer/C21_embedding/C21_embedding_weights/strided_slice/stack_1dinput_layer/sparse_input_layer/input_layer/C21_embedding/C21_embedding_weights/strided_slice/stack_2*
Index0*
shrink_axis_mask*
_output_shapes
: *
T0
Ш
Vinput_layer/sparse_input_layer/input_layer/C21_embedding/C21_embedding_weights/stack/0Const*
dtype0*
value	B :*
_output_shapes
: 
ј
Tinput_layer/sparse_input_layer/input_layer/C21_embedding/C21_embedding_weights/stackPackVinput_layer/sparse_input_layer/input_layer/C21_embedding/C21_embedding_weights/stack/0\input_layer/sparse_input_layer/input_layer/C21_embedding/C21_embedding_weights/strided_slice*
_output_shapes
:*
N*
T0
∆
Sinput_layer/sparse_input_layer/input_layer/C21_embedding/C21_embedding_weights/TileTileXinput_layer/sparse_input_layer/input_layer/C21_embedding/C21_embedding_weights/Reshape_1Tinput_layer/sparse_input_layer/input_layer/C21_embedding/C21_embedding_weights/stack*
T0
*0
_output_shapes
:€€€€€€€€€€€€€€€€€€
А
Yinput_layer/sparse_input_layer/input_layer/C21_embedding/C21_embedding_weights/zeros_like	ZerosLikefinput_layer/sparse_input_layer/input_layer/C21_embedding/C21_embedding_weights/embedding_lookup_sparse*
T0*'
_output_shapes
:€€€€€€€€€
Ґ
Ninput_layer/sparse_input_layer/input_layer/C21_embedding/C21_embedding_weightsSelectSinput_layer/sparse_input_layer/input_layer/C21_embedding/C21_embedding_weights/TileYinput_layer/sparse_input_layer/input_layer/C21_embedding/C21_embedding_weights/zeros_likefinput_layer/sparse_input_layer/input_layer/C21_embedding/C21_embedding_weights/embedding_lookup_sparse*'
_output_shapes
:€€€€€€€€€*
T0
з
Uinput_layer/sparse_input_layer/input_layer/C21_embedding/C21_embedding_weights/Cast_1CastTinput_layer/sparse_input_layer/input_layer/C21_embedding/to_sparse_input/dense_shape*

DstT0*
_output_shapes
:*

SrcT0	
¶
\input_layer/sparse_input_layer/input_layer/C21_embedding/C21_embedding_weights/Slice_1/beginConst*
_output_shapes
:*
valueB: *
dtype0
•
[input_layer/sparse_input_layer/input_layer/C21_embedding/C21_embedding_weights/Slice_1/sizeConst*
valueB:*
_output_shapes
:*
dtype0
£
Vinput_layer/sparse_input_layer/input_layer/C21_embedding/C21_embedding_weights/Slice_1SliceUinput_layer/sparse_input_layer/input_layer/C21_embedding/C21_embedding_weights/Cast_1\input_layer/sparse_input_layer/input_layer/C21_embedding/C21_embedding_weights/Slice_1/begin[input_layer/sparse_input_layer/input_layer/C21_embedding/C21_embedding_weights/Slice_1/size*
_output_shapes
:*
Index0*
T0
‘
Vinput_layer/sparse_input_layer/input_layer/C21_embedding/C21_embedding_weights/Shape_1ShapeNinput_layer/sparse_input_layer/input_layer/C21_embedding/C21_embedding_weights*
T0*
_output_shapes
:
¶
\input_layer/sparse_input_layer/input_layer/C21_embedding/C21_embedding_weights/Slice_2/beginConst*
valueB:*
_output_shapes
:*
dtype0
Ѓ
[input_layer/sparse_input_layer/input_layer/C21_embedding/C21_embedding_weights/Slice_2/sizeConst*
valueB:
€€€€€€€€€*
dtype0*
_output_shapes
:
§
Vinput_layer/sparse_input_layer/input_layer/C21_embedding/C21_embedding_weights/Slice_2SliceVinput_layer/sparse_input_layer/input_layer/C21_embedding/C21_embedding_weights/Shape_1\input_layer/sparse_input_layer/input_layer/C21_embedding/C21_embedding_weights/Slice_2/begin[input_layer/sparse_input_layer/input_layer/C21_embedding/C21_embedding_weights/Slice_2/size*
T0*
_output_shapes
:*
Index0
Ь
Zinput_layer/sparse_input_layer/input_layer/C21_embedding/C21_embedding_weights/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
Ы
Uinput_layer/sparse_input_layer/input_layer/C21_embedding/C21_embedding_weights/concatConcatV2Vinput_layer/sparse_input_layer/input_layer/C21_embedding/C21_embedding_weights/Slice_1Vinput_layer/sparse_input_layer/input_layer/C21_embedding/C21_embedding_weights/Slice_2Zinput_layer/sparse_input_layer/input_layer/C21_embedding/C21_embedding_weights/concat/axis*
T0*
_output_shapes
:*
N
Љ
Xinput_layer/sparse_input_layer/input_layer/C21_embedding/C21_embedding_weights/Reshape_2ReshapeNinput_layer/sparse_input_layer/input_layer/C21_embedding/C21_embedding_weightsUinput_layer/sparse_input_layer/input_layer/C21_embedding/C21_embedding_weights/concat*'
_output_shapes
:€€€€€€€€€*
T0
∆
>input_layer/sparse_input_layer/input_layer/C21_embedding/ShapeShapeXinput_layer/sparse_input_layer/input_layer/C21_embedding/C21_embedding_weights/Reshape_2*
_output_shapes
:*
T0
Ц
Linput_layer/sparse_input_layer/input_layer/C21_embedding/strided_slice/stackConst*
valueB: *
_output_shapes
:*
dtype0
Ш
Ninput_layer/sparse_input_layer/input_layer/C21_embedding/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
Ш
Ninput_layer/sparse_input_layer/input_layer/C21_embedding/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
 
Finput_layer/sparse_input_layer/input_layer/C21_embedding/strided_sliceStridedSlice>input_layer/sparse_input_layer/input_layer/C21_embedding/ShapeLinput_layer/sparse_input_layer/input_layer/C21_embedding/strided_slice/stackNinput_layer/sparse_input_layer/input_layer/C21_embedding/strided_slice/stack_1Ninput_layer/sparse_input_layer/input_layer/C21_embedding/strided_slice/stack_2*
T0*
Index0*
_output_shapes
: *
shrink_axis_mask
К
Hinput_layer/sparse_input_layer/input_layer/C21_embedding/Reshape/shape/1Const*
value	B :*
_output_shapes
: *
dtype0
О
Finput_layer/sparse_input_layer/input_layer/C21_embedding/Reshape/shapePackFinput_layer/sparse_input_layer/input_layer/C21_embedding/strided_sliceHinput_layer/sparse_input_layer/input_layer/C21_embedding/Reshape/shape/1*
_output_shapes
:*
T0*
N
Я
@input_layer/sparse_input_layer/input_layer/C21_embedding/ReshapeReshapeXinput_layer/sparse_input_layer/input_layer/C21_embedding/C21_embedding_weights/Reshape_2Finput_layer/sparse_input_layer/input_layer/C21_embedding/Reshape/shape*'
_output_shapes
:€€€€€€€€€*
T0
Т
Ginput_layer/sparse_input_layer/input_layer/C22_embedding/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€
№
Cinput_layer/sparse_input_layer/input_layer/C22_embedding/ExpandDims
ExpandDimsPlaceholder_34Ginput_layer/sparse_input_layer/input_layer/C22_embedding/ExpandDims/dim*
T0*'
_output_shapes
:€€€€€€€€€
Ш
Winput_layer/sparse_input_layer/input_layer/C22_embedding/to_sparse_input/ignore_value/xConst*
valueB B *
_output_shapes
: *
dtype0
≠
Qinput_layer/sparse_input_layer/input_layer/C22_embedding/to_sparse_input/NotEqualNotEqualCinput_layer/sparse_input_layer/input_layer/C22_embedding/ExpandDimsWinput_layer/sparse_input_layer/input_layer/C22_embedding/to_sparse_input/ignore_value/x*
T0*'
_output_shapes
:€€€€€€€€€
’
Pinput_layer/sparse_input_layer/input_layer/C22_embedding/to_sparse_input/indicesWhereQinput_layer/sparse_input_layer/input_layer/C22_embedding/to_sparse_input/NotEqual*'
_output_shapes
:€€€€€€€€€
ґ
Oinput_layer/sparse_input_layer/input_layer/C22_embedding/to_sparse_input/valuesGatherNdCinput_layer/sparse_input_layer/input_layer/C22_embedding/ExpandDimsPinput_layer/sparse_input_layer/input_layer/C22_embedding/to_sparse_input/indices*
Tparams0*
Tindices0	*#
_output_shapes
:€€€€€€€€€
„
Tinput_layer/sparse_input_layer/input_layer/C22_embedding/to_sparse_input/dense_shapeShapeCinput_layer/sparse_input_layer/input_layer/C22_embedding/ExpandDims*
out_type0	*
_output_shapes
:*
T0
г
?input_layer/sparse_input_layer/input_layer/C22_embedding/lookupStringToHashBucketFastOinput_layer/sparse_input_layer/input_layer/C22_embedding/to_sparse_input/values*#
_output_shapes
:€€€€€€€€€*
num_bucketsРN
Э
minput_layer/sparse_input_layer/input_layer/C22_embedding/embedding_weights/Initializer/truncated_normal/shapeConst*]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C22_embedding/embedding_weights*
_output_shapes
:*
valueB"'     *
dtype0
Р
linput_layer/sparse_input_layer/input_layer/C22_embedding/embedding_weights/Initializer/truncated_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    *]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C22_embedding/embedding_weights
Т
ninput_layer/sparse_input_layer/input_layer/C22_embedding/embedding_weights/Initializer/truncated_normal/stddevConst*]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C22_embedding/embedding_weights*
dtype0*
valueB
 *  А>*
_output_shapes
: 
П
winput_layer/sparse_input_layer/input_layer/C22_embedding/embedding_weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormalminput_layer/sparse_input_layer/input_layer/C22_embedding/embedding_weights/Initializer/truncated_normal/shape*
T0*
_output_shapes
:	РN*
dtype0*]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C22_embedding/embedding_weights
д
kinput_layer/sparse_input_layer/input_layer/C22_embedding/embedding_weights/Initializer/truncated_normal/mulMulwinput_layer/sparse_input_layer/input_layer/C22_embedding/embedding_weights/Initializer/truncated_normal/TruncatedNormalninput_layer/sparse_input_layer/input_layer/C22_embedding/embedding_weights/Initializer/truncated_normal/stddev*]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C22_embedding/embedding_weights*
T0*
_output_shapes
:	РN
“
ginput_layer/sparse_input_layer/input_layer/C22_embedding/embedding_weights/Initializer/truncated_normalAddkinput_layer/sparse_input_layer/input_layer/C22_embedding/embedding_weights/Initializer/truncated_normal/mullinput_layer/sparse_input_layer/input_layer/C22_embedding/embedding_weights/Initializer/truncated_normal/mean*
_output_shapes
:	РN*
T0*]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C22_embedding/embedding_weights
ы
Jinput_layer/sparse_input_layer/input_layer/C22_embedding/embedding_weights
VariableV2*
shape:	РN*]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C22_embedding/embedding_weights*
_output_shapes
:	РN*
dtype0
Щ
Qinput_layer/sparse_input_layer/input_layer/C22_embedding/embedding_weights/AssignAssignJinput_layer/sparse_input_layer/input_layer/C22_embedding/embedding_weightsginput_layer/sparse_input_layer/input_layer/C22_embedding/embedding_weights/Initializer/truncated_normal*
_output_shapes
:	РN*]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C22_embedding/embedding_weights*
T0
∞
Oinput_layer/sparse_input_layer/input_layer/C22_embedding/embedding_weights/readIdentityJinput_layer/sparse_input_layer/input_layer/C22_embedding/embedding_weights*
_output_shapes
:	РN*]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C22_embedding/embedding_weights*
T0
§
Zinput_layer/sparse_input_layer/input_layer/C22_embedding/C22_embedding_weights/Slice/beginConst*
dtype0*
_output_shapes
:*
valueB: 
£
Yinput_layer/sparse_input_layer/input_layer/C22_embedding/C22_embedding_weights/Slice/sizeConst*
valueB:*
dtype0*
_output_shapes
:
Ь
Tinput_layer/sparse_input_layer/input_layer/C22_embedding/C22_embedding_weights/SliceSliceTinput_layer/sparse_input_layer/input_layer/C22_embedding/to_sparse_input/dense_shapeZinput_layer/sparse_input_layer/input_layer/C22_embedding/C22_embedding_weights/Slice/beginYinput_layer/sparse_input_layer/input_layer/C22_embedding/C22_embedding_weights/Slice/size*
_output_shapes
:*
Index0*
T0	
Ю
Tinput_layer/sparse_input_layer/input_layer/C22_embedding/C22_embedding_weights/ConstConst*
_output_shapes
:*
valueB: *
dtype0
®
Sinput_layer/sparse_input_layer/input_layer/C22_embedding/C22_embedding_weights/ProdProdTinput_layer/sparse_input_layer/input_layer/C22_embedding/C22_embedding_weights/SliceTinput_layer/sparse_input_layer/input_layer/C22_embedding/C22_embedding_weights/Const*
T0	*
_output_shapes
: 
°
_input_layer/sparse_input_layer/input_layer/C22_embedding/C22_embedding_weights/GatherV2/indicesConst*
_output_shapes
: *
dtype0*
value	B :
Ю
\input_layer/sparse_input_layer/input_layer/C22_embedding/C22_embedding_weights/GatherV2/axisConst*
value	B : *
dtype0*
_output_shapes
: 
Љ
Winput_layer/sparse_input_layer/input_layer/C22_embedding/C22_embedding_weights/GatherV2GatherV2Tinput_layer/sparse_input_layer/input_layer/C22_embedding/to_sparse_input/dense_shape_input_layer/sparse_input_layer/input_layer/C22_embedding/C22_embedding_weights/GatherV2/indices\input_layer/sparse_input_layer/input_layer/C22_embedding/C22_embedding_weights/GatherV2/axis*
Tindices0*
_output_shapes
: *
Taxis0*
Tparams0	
є
Uinput_layer/sparse_input_layer/input_layer/C22_embedding/C22_embedding_weights/Cast/xPackSinput_layer/sparse_input_layer/input_layer/C22_embedding/C22_embedding_weights/ProdWinput_layer/sparse_input_layer/input_layer/C22_embedding/C22_embedding_weights/GatherV2*
N*
_output_shapes
:*
T0	
Ы
\input_layer/sparse_input_layer/input_layer/C22_embedding/C22_embedding_weights/SparseReshapeSparseReshapePinput_layer/sparse_input_layer/input_layer/C22_embedding/to_sparse_input/indicesTinput_layer/sparse_input_layer/input_layer/C22_embedding/to_sparse_input/dense_shapeUinput_layer/sparse_input_layer/input_layer/C22_embedding/C22_embedding_weights/Cast/x*-
_output_shapes
:€€€€€€€€€:
а
einput_layer/sparse_input_layer/input_layer/C22_embedding/C22_embedding_weights/SparseReshape/IdentityIdentity?input_layer/sparse_input_layer/input_layer/C22_embedding/lookup*
T0	*#
_output_shapes
:€€€€€€€€€
Я
]input_layer/sparse_input_layer/input_layer/C22_embedding/C22_embedding_weights/GreaterEqual/yConst*
dtype0	*
value	B	 R *
_output_shapes
: 
я
[input_layer/sparse_input_layer/input_layer/C22_embedding/C22_embedding_weights/GreaterEqualGreaterEqualeinput_layer/sparse_input_layer/input_layer/C22_embedding/C22_embedding_weights/SparseReshape/Identity]input_layer/sparse_input_layer/input_layer/C22_embedding/C22_embedding_weights/GreaterEqual/y*
T0	*#
_output_shapes
:€€€€€€€€€
г
Tinput_layer/sparse_input_layer/input_layer/C22_embedding/C22_embedding_weights/WhereWhere[input_layer/sparse_input_layer/input_layer/C22_embedding/C22_embedding_weights/GreaterEqual*'
_output_shapes
:€€€€€€€€€
ѓ
\input_layer/sparse_input_layer/input_layer/C22_embedding/C22_embedding_weights/Reshape/shapeConst*
dtype0*
_output_shapes
:*
valueB:
€€€€€€€€€
√
Vinput_layer/sparse_input_layer/input_layer/C22_embedding/C22_embedding_weights/ReshapeReshapeTinput_layer/sparse_input_layer/input_layer/C22_embedding/C22_embedding_weights/Where\input_layer/sparse_input_layer/input_layer/C22_embedding/C22_embedding_weights/Reshape/shape*
T0	*#
_output_shapes
:€€€€€€€€€
†
^input_layer/sparse_input_layer/input_layer/C22_embedding/C22_embedding_weights/GatherV2_1/axisConst*
_output_shapes
: *
value	B : *
dtype0
–
Yinput_layer/sparse_input_layer/input_layer/C22_embedding/C22_embedding_weights/GatherV2_1GatherV2\input_layer/sparse_input_layer/input_layer/C22_embedding/C22_embedding_weights/SparseReshapeVinput_layer/sparse_input_layer/input_layer/C22_embedding/C22_embedding_weights/Reshape^input_layer/sparse_input_layer/input_layer/C22_embedding/C22_embedding_weights/GatherV2_1/axis*'
_output_shapes
:€€€€€€€€€*
Tindices0	*
Tparams0	*
Taxis0
†
^input_layer/sparse_input_layer/input_layer/C22_embedding/C22_embedding_weights/GatherV2_2/axisConst*
dtype0*
value	B : *
_output_shapes
: 
’
Yinput_layer/sparse_input_layer/input_layer/C22_embedding/C22_embedding_weights/GatherV2_2GatherV2einput_layer/sparse_input_layer/input_layer/C22_embedding/C22_embedding_weights/SparseReshape/IdentityVinput_layer/sparse_input_layer/input_layer/C22_embedding/C22_embedding_weights/Reshape^input_layer/sparse_input_layer/input_layer/C22_embedding/C22_embedding_weights/GatherV2_2/axis*
Tparams0	*
Tindices0	*
Taxis0*#
_output_shapes
:€€€€€€€€€
и
Winput_layer/sparse_input_layer/input_layer/C22_embedding/C22_embedding_weights/IdentityIdentity^input_layer/sparse_input_layer/input_layer/C22_embedding/C22_embedding_weights/SparseReshape:1*
T0	*
_output_shapes
:
™
hinput_layer/sparse_input_layer/input_layer/C22_embedding/C22_embedding_weights/SparseFillEmptyRows/ConstConst*
_output_shapes
: *
value	B	 R *
dtype0	
е
vinput_layer/sparse_input_layer/input_layer/C22_embedding/C22_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowsSparseFillEmptyRowsYinput_layer/sparse_input_layer/input_layer/C22_embedding/C22_embedding_weights/GatherV2_1Yinput_layer/sparse_input_layer/input_layer/C22_embedding/C22_embedding_weights/GatherV2_2Winput_layer/sparse_input_layer/input_layer/C22_embedding/C22_embedding_weights/Identityhinput_layer/sparse_input_layer/input_layer/C22_embedding/C22_embedding_weights/SparseFillEmptyRows/Const*
T0	*T
_output_shapesB
@:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€
Ћ
zinput_layer/sparse_input_layer/input_layer/C22_embedding/C22_embedding_weights/embedding_lookup_sparse/strided_slice/stackConst*
valueB"        *
_output_shapes
:*
dtype0
Ќ
|input_layer/sparse_input_layer/input_layer/C22_embedding/C22_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1Const*
dtype0*
valueB"       *
_output_shapes
:
Ќ
|input_layer/sparse_input_layer/input_layer/C22_embedding/C22_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:
й
tinput_layer/sparse_input_layer/input_layer/C22_embedding/C22_embedding_weights/embedding_lookup_sparse/strided_sliceStridedSlicevinput_layer/sparse_input_layer/input_layer/C22_embedding/C22_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowszinput_layer/sparse_input_layer/input_layer/C22_embedding/C22_embedding_weights/embedding_lookup_sparse/strided_slice/stack|input_layer/sparse_input_layer/input_layer/C22_embedding/C22_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1|input_layer/sparse_input_layer/input_layer/C22_embedding/C22_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2*#
_output_shapes
:€€€€€€€€€*
end_mask*

begin_mask*
T0	*
shrink_axis_mask*
Index0
¶
kinput_layer/sparse_input_layer/input_layer/C22_embedding/C22_embedding_weights/embedding_lookup_sparse/CastCasttinput_layer/sparse_input_layer/input_layer/C22_embedding/C22_embedding_weights/embedding_lookup_sparse/strided_slice*

DstT0*

SrcT0	*#
_output_shapes
:€€€€€€€€€
Ѓ
minput_layer/sparse_input_layer/input_layer/C22_embedding/C22_embedding_weights/embedding_lookup_sparse/UniqueUniquexinput_layer/sparse_input_layer/input_layer/C22_embedding/C22_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€*
T0	
Э
|input_layer/sparse_input_layer/input_layer/C22_embedding/C22_embedding_weights/embedding_lookup_sparse/embedding_lookup/axisConst*
dtype0*
_output_shapes
: *]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C22_embedding/embedding_weights*
value	B : 
х
winput_layer/sparse_input_layer/input_layer/C22_embedding/C22_embedding_weights/embedding_lookup_sparse/embedding_lookupGatherV2Oinput_layer/sparse_input_layer/input_layer/C22_embedding/embedding_weights/readminput_layer/sparse_input_layer/input_layer/C22_embedding/C22_embedding_weights/embedding_lookup_sparse/Unique|input_layer/sparse_input_layer/input_layer/C22_embedding/C22_embedding_weights/embedding_lookup_sparse/embedding_lookup/axis*
Tindices0	*
Tparams0*]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C22_embedding/embedding_weights*
Taxis0*'
_output_shapes
:€€€€€€€€€
Є
Аinput_layer/sparse_input_layer/input_layer/C22_embedding/C22_embedding_weights/embedding_lookup_sparse/embedding_lookup/IdentityIdentitywinput_layer/sparse_input_layer/input_layer/C22_embedding/C22_embedding_weights/embedding_lookup_sparse/embedding_lookup*
T0*'
_output_shapes
:€€€€€€€€€
О
finput_layer/sparse_input_layer/input_layer/C22_embedding/C22_embedding_weights/embedding_lookup_sparseSparseSegmentMeanАinput_layer/sparse_input_layer/input_layer/C22_embedding/C22_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identityoinput_layer/sparse_input_layer/input_layer/C22_embedding/C22_embedding_weights/embedding_lookup_sparse/Unique:1kinput_layer/sparse_input_layer/input_layer/C22_embedding/C22_embedding_weights/embedding_lookup_sparse/Cast*
T0*'
_output_shapes
:€€€€€€€€€
ѓ
^input_layer/sparse_input_layer/input_layer/C22_embedding/C22_embedding_weights/Reshape_1/shapeConst*
valueB"€€€€   *
_output_shapes
:*
dtype0
п
Xinput_layer/sparse_input_layer/input_layer/C22_embedding/C22_embedding_weights/Reshape_1Reshapexinput_layer/sparse_input_layer/input_layer/C22_embedding/C22_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:2^input_layer/sparse_input_layer/input_layer/C22_embedding/C22_embedding_weights/Reshape_1/shape*'
_output_shapes
:€€€€€€€€€*
T0

к
Tinput_layer/sparse_input_layer/input_layer/C22_embedding/C22_embedding_weights/ShapeShapefinput_layer/sparse_input_layer/input_layer/C22_embedding/C22_embedding_weights/embedding_lookup_sparse*
T0*
_output_shapes
:
ђ
binput_layer/sparse_input_layer/input_layer/C22_embedding/C22_embedding_weights/strided_slice/stackConst*
valueB:*
_output_shapes
:*
dtype0
Ѓ
dinput_layer/sparse_input_layer/input_layer/C22_embedding/C22_embedding_weights/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
Ѓ
dinput_layer/sparse_input_layer/input_layer/C22_embedding/C22_embedding_weights/strided_slice/stack_2Const*
_output_shapes
:*
valueB:*
dtype0
Є
\input_layer/sparse_input_layer/input_layer/C22_embedding/C22_embedding_weights/strided_sliceStridedSliceTinput_layer/sparse_input_layer/input_layer/C22_embedding/C22_embedding_weights/Shapebinput_layer/sparse_input_layer/input_layer/C22_embedding/C22_embedding_weights/strided_slice/stackdinput_layer/sparse_input_layer/input_layer/C22_embedding/C22_embedding_weights/strided_slice/stack_1dinput_layer/sparse_input_layer/input_layer/C22_embedding/C22_embedding_weights/strided_slice/stack_2*
_output_shapes
: *
shrink_axis_mask*
T0*
Index0
Ш
Vinput_layer/sparse_input_layer/input_layer/C22_embedding/C22_embedding_weights/stack/0Const*
value	B :*
_output_shapes
: *
dtype0
ј
Tinput_layer/sparse_input_layer/input_layer/C22_embedding/C22_embedding_weights/stackPackVinput_layer/sparse_input_layer/input_layer/C22_embedding/C22_embedding_weights/stack/0\input_layer/sparse_input_layer/input_layer/C22_embedding/C22_embedding_weights/strided_slice*
_output_shapes
:*
N*
T0
∆
Sinput_layer/sparse_input_layer/input_layer/C22_embedding/C22_embedding_weights/TileTileXinput_layer/sparse_input_layer/input_layer/C22_embedding/C22_embedding_weights/Reshape_1Tinput_layer/sparse_input_layer/input_layer/C22_embedding/C22_embedding_weights/stack*
T0
*0
_output_shapes
:€€€€€€€€€€€€€€€€€€
А
Yinput_layer/sparse_input_layer/input_layer/C22_embedding/C22_embedding_weights/zeros_like	ZerosLikefinput_layer/sparse_input_layer/input_layer/C22_embedding/C22_embedding_weights/embedding_lookup_sparse*'
_output_shapes
:€€€€€€€€€*
T0
Ґ
Ninput_layer/sparse_input_layer/input_layer/C22_embedding/C22_embedding_weightsSelectSinput_layer/sparse_input_layer/input_layer/C22_embedding/C22_embedding_weights/TileYinput_layer/sparse_input_layer/input_layer/C22_embedding/C22_embedding_weights/zeros_likefinput_layer/sparse_input_layer/input_layer/C22_embedding/C22_embedding_weights/embedding_lookup_sparse*'
_output_shapes
:€€€€€€€€€*
T0
з
Uinput_layer/sparse_input_layer/input_layer/C22_embedding/C22_embedding_weights/Cast_1CastTinput_layer/sparse_input_layer/input_layer/C22_embedding/to_sparse_input/dense_shape*

SrcT0	*

DstT0*
_output_shapes
:
¶
\input_layer/sparse_input_layer/input_layer/C22_embedding/C22_embedding_weights/Slice_1/beginConst*
dtype0*
valueB: *
_output_shapes
:
•
[input_layer/sparse_input_layer/input_layer/C22_embedding/C22_embedding_weights/Slice_1/sizeConst*
valueB:*
dtype0*
_output_shapes
:
£
Vinput_layer/sparse_input_layer/input_layer/C22_embedding/C22_embedding_weights/Slice_1SliceUinput_layer/sparse_input_layer/input_layer/C22_embedding/C22_embedding_weights/Cast_1\input_layer/sparse_input_layer/input_layer/C22_embedding/C22_embedding_weights/Slice_1/begin[input_layer/sparse_input_layer/input_layer/C22_embedding/C22_embedding_weights/Slice_1/size*
T0*
_output_shapes
:*
Index0
‘
Vinput_layer/sparse_input_layer/input_layer/C22_embedding/C22_embedding_weights/Shape_1ShapeNinput_layer/sparse_input_layer/input_layer/C22_embedding/C22_embedding_weights*
_output_shapes
:*
T0
¶
\input_layer/sparse_input_layer/input_layer/C22_embedding/C22_embedding_weights/Slice_2/beginConst*
dtype0*
valueB:*
_output_shapes
:
Ѓ
[input_layer/sparse_input_layer/input_layer/C22_embedding/C22_embedding_weights/Slice_2/sizeConst*
dtype0*
valueB:
€€€€€€€€€*
_output_shapes
:
§
Vinput_layer/sparse_input_layer/input_layer/C22_embedding/C22_embedding_weights/Slice_2SliceVinput_layer/sparse_input_layer/input_layer/C22_embedding/C22_embedding_weights/Shape_1\input_layer/sparse_input_layer/input_layer/C22_embedding/C22_embedding_weights/Slice_2/begin[input_layer/sparse_input_layer/input_layer/C22_embedding/C22_embedding_weights/Slice_2/size*
_output_shapes
:*
Index0*
T0
Ь
Zinput_layer/sparse_input_layer/input_layer/C22_embedding/C22_embedding_weights/concat/axisConst*
dtype0*
value	B : *
_output_shapes
: 
Ы
Uinput_layer/sparse_input_layer/input_layer/C22_embedding/C22_embedding_weights/concatConcatV2Vinput_layer/sparse_input_layer/input_layer/C22_embedding/C22_embedding_weights/Slice_1Vinput_layer/sparse_input_layer/input_layer/C22_embedding/C22_embedding_weights/Slice_2Zinput_layer/sparse_input_layer/input_layer/C22_embedding/C22_embedding_weights/concat/axis*
_output_shapes
:*
T0*
N
Љ
Xinput_layer/sparse_input_layer/input_layer/C22_embedding/C22_embedding_weights/Reshape_2ReshapeNinput_layer/sparse_input_layer/input_layer/C22_embedding/C22_embedding_weightsUinput_layer/sparse_input_layer/input_layer/C22_embedding/C22_embedding_weights/concat*'
_output_shapes
:€€€€€€€€€*
T0
∆
>input_layer/sparse_input_layer/input_layer/C22_embedding/ShapeShapeXinput_layer/sparse_input_layer/input_layer/C22_embedding/C22_embedding_weights/Reshape_2*
_output_shapes
:*
T0
Ц
Linput_layer/sparse_input_layer/input_layer/C22_embedding/strided_slice/stackConst*
dtype0*
valueB: *
_output_shapes
:
Ш
Ninput_layer/sparse_input_layer/input_layer/C22_embedding/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Ш
Ninput_layer/sparse_input_layer/input_layer/C22_embedding/strided_slice/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
 
Finput_layer/sparse_input_layer/input_layer/C22_embedding/strided_sliceStridedSlice>input_layer/sparse_input_layer/input_layer/C22_embedding/ShapeLinput_layer/sparse_input_layer/input_layer/C22_embedding/strided_slice/stackNinput_layer/sparse_input_layer/input_layer/C22_embedding/strided_slice/stack_1Ninput_layer/sparse_input_layer/input_layer/C22_embedding/strided_slice/stack_2*
T0*
_output_shapes
: *
Index0*
shrink_axis_mask
К
Hinput_layer/sparse_input_layer/input_layer/C22_embedding/Reshape/shape/1Const*
dtype0*
value	B :*
_output_shapes
: 
О
Finput_layer/sparse_input_layer/input_layer/C22_embedding/Reshape/shapePackFinput_layer/sparse_input_layer/input_layer/C22_embedding/strided_sliceHinput_layer/sparse_input_layer/input_layer/C22_embedding/Reshape/shape/1*
_output_shapes
:*
N*
T0
Я
@input_layer/sparse_input_layer/input_layer/C22_embedding/ReshapeReshapeXinput_layer/sparse_input_layer/input_layer/C22_embedding/C22_embedding_weights/Reshape_2Finput_layer/sparse_input_layer/input_layer/C22_embedding/Reshape/shape*'
_output_shapes
:€€€€€€€€€*
T0
Т
Ginput_layer/sparse_input_layer/input_layer/C23_embedding/ExpandDims/dimConst*
_output_shapes
: *
valueB :
€€€€€€€€€*
dtype0
№
Cinput_layer/sparse_input_layer/input_layer/C23_embedding/ExpandDims
ExpandDimsPlaceholder_35Ginput_layer/sparse_input_layer/input_layer/C23_embedding/ExpandDims/dim*
T0*'
_output_shapes
:€€€€€€€€€
Ш
Winput_layer/sparse_input_layer/input_layer/C23_embedding/to_sparse_input/ignore_value/xConst*
dtype0*
_output_shapes
: *
valueB B 
≠
Qinput_layer/sparse_input_layer/input_layer/C23_embedding/to_sparse_input/NotEqualNotEqualCinput_layer/sparse_input_layer/input_layer/C23_embedding/ExpandDimsWinput_layer/sparse_input_layer/input_layer/C23_embedding/to_sparse_input/ignore_value/x*'
_output_shapes
:€€€€€€€€€*
T0
’
Pinput_layer/sparse_input_layer/input_layer/C23_embedding/to_sparse_input/indicesWhereQinput_layer/sparse_input_layer/input_layer/C23_embedding/to_sparse_input/NotEqual*'
_output_shapes
:€€€€€€€€€
ґ
Oinput_layer/sparse_input_layer/input_layer/C23_embedding/to_sparse_input/valuesGatherNdCinput_layer/sparse_input_layer/input_layer/C23_embedding/ExpandDimsPinput_layer/sparse_input_layer/input_layer/C23_embedding/to_sparse_input/indices*
Tparams0*
Tindices0	*#
_output_shapes
:€€€€€€€€€
„
Tinput_layer/sparse_input_layer/input_layer/C23_embedding/to_sparse_input/dense_shapeShapeCinput_layer/sparse_input_layer/input_layer/C23_embedding/ExpandDims*
out_type0	*
_output_shapes
:*
T0
г
?input_layer/sparse_input_layer/input_layer/C23_embedding/lookupStringToHashBucketFastOinput_layer/sparse_input_layer/input_layer/C23_embedding/to_sparse_input/values*
num_bucketsРN*#
_output_shapes
:€€€€€€€€€
Э
minput_layer/sparse_input_layer/input_layer/C23_embedding/embedding_weights/Initializer/truncated_normal/shapeConst*
_output_shapes
:*
dtype0*
valueB"'     *]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C23_embedding/embedding_weights
Р
linput_layer/sparse_input_layer/input_layer/C23_embedding/embedding_weights/Initializer/truncated_normal/meanConst*
dtype0*]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C23_embedding/embedding_weights*
valueB
 *    *
_output_shapes
: 
Т
ninput_layer/sparse_input_layer/input_layer/C23_embedding/embedding_weights/Initializer/truncated_normal/stddevConst*]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C23_embedding/embedding_weights*
dtype0*
valueB
 *  А>*
_output_shapes
: 
П
winput_layer/sparse_input_layer/input_layer/C23_embedding/embedding_weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormalminput_layer/sparse_input_layer/input_layer/C23_embedding/embedding_weights/Initializer/truncated_normal/shape*
T0*
_output_shapes
:	РN*
dtype0*]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C23_embedding/embedding_weights
д
kinput_layer/sparse_input_layer/input_layer/C23_embedding/embedding_weights/Initializer/truncated_normal/mulMulwinput_layer/sparse_input_layer/input_layer/C23_embedding/embedding_weights/Initializer/truncated_normal/TruncatedNormalninput_layer/sparse_input_layer/input_layer/C23_embedding/embedding_weights/Initializer/truncated_normal/stddev*]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C23_embedding/embedding_weights*
T0*
_output_shapes
:	РN
“
ginput_layer/sparse_input_layer/input_layer/C23_embedding/embedding_weights/Initializer/truncated_normalAddkinput_layer/sparse_input_layer/input_layer/C23_embedding/embedding_weights/Initializer/truncated_normal/mullinput_layer/sparse_input_layer/input_layer/C23_embedding/embedding_weights/Initializer/truncated_normal/mean*
_output_shapes
:	РN*
T0*]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C23_embedding/embedding_weights
ы
Jinput_layer/sparse_input_layer/input_layer/C23_embedding/embedding_weights
VariableV2*
_output_shapes
:	РN*]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C23_embedding/embedding_weights*
shape:	РN*
dtype0
Щ
Qinput_layer/sparse_input_layer/input_layer/C23_embedding/embedding_weights/AssignAssignJinput_layer/sparse_input_layer/input_layer/C23_embedding/embedding_weightsginput_layer/sparse_input_layer/input_layer/C23_embedding/embedding_weights/Initializer/truncated_normal*]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C23_embedding/embedding_weights*
T0*
_output_shapes
:	РN
∞
Oinput_layer/sparse_input_layer/input_layer/C23_embedding/embedding_weights/readIdentityJinput_layer/sparse_input_layer/input_layer/C23_embedding/embedding_weights*]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C23_embedding/embedding_weights*
_output_shapes
:	РN*
T0
§
Zinput_layer/sparse_input_layer/input_layer/C23_embedding/C23_embedding_weights/Slice/beginConst*
_output_shapes
:*
dtype0*
valueB: 
£
Yinput_layer/sparse_input_layer/input_layer/C23_embedding/C23_embedding_weights/Slice/sizeConst*
dtype0*
valueB:*
_output_shapes
:
Ь
Tinput_layer/sparse_input_layer/input_layer/C23_embedding/C23_embedding_weights/SliceSliceTinput_layer/sparse_input_layer/input_layer/C23_embedding/to_sparse_input/dense_shapeZinput_layer/sparse_input_layer/input_layer/C23_embedding/C23_embedding_weights/Slice/beginYinput_layer/sparse_input_layer/input_layer/C23_embedding/C23_embedding_weights/Slice/size*
_output_shapes
:*
T0	*
Index0
Ю
Tinput_layer/sparse_input_layer/input_layer/C23_embedding/C23_embedding_weights/ConstConst*
_output_shapes
:*
valueB: *
dtype0
®
Sinput_layer/sparse_input_layer/input_layer/C23_embedding/C23_embedding_weights/ProdProdTinput_layer/sparse_input_layer/input_layer/C23_embedding/C23_embedding_weights/SliceTinput_layer/sparse_input_layer/input_layer/C23_embedding/C23_embedding_weights/Const*
T0	*
_output_shapes
: 
°
_input_layer/sparse_input_layer/input_layer/C23_embedding/C23_embedding_weights/GatherV2/indicesConst*
value	B :*
dtype0*
_output_shapes
: 
Ю
\input_layer/sparse_input_layer/input_layer/C23_embedding/C23_embedding_weights/GatherV2/axisConst*
value	B : *
dtype0*
_output_shapes
: 
Љ
Winput_layer/sparse_input_layer/input_layer/C23_embedding/C23_embedding_weights/GatherV2GatherV2Tinput_layer/sparse_input_layer/input_layer/C23_embedding/to_sparse_input/dense_shape_input_layer/sparse_input_layer/input_layer/C23_embedding/C23_embedding_weights/GatherV2/indices\input_layer/sparse_input_layer/input_layer/C23_embedding/C23_embedding_weights/GatherV2/axis*
Taxis0*
_output_shapes
: *
Tparams0	*
Tindices0
є
Uinput_layer/sparse_input_layer/input_layer/C23_embedding/C23_embedding_weights/Cast/xPackSinput_layer/sparse_input_layer/input_layer/C23_embedding/C23_embedding_weights/ProdWinput_layer/sparse_input_layer/input_layer/C23_embedding/C23_embedding_weights/GatherV2*
N*
T0	*
_output_shapes
:
Ы
\input_layer/sparse_input_layer/input_layer/C23_embedding/C23_embedding_weights/SparseReshapeSparseReshapePinput_layer/sparse_input_layer/input_layer/C23_embedding/to_sparse_input/indicesTinput_layer/sparse_input_layer/input_layer/C23_embedding/to_sparse_input/dense_shapeUinput_layer/sparse_input_layer/input_layer/C23_embedding/C23_embedding_weights/Cast/x*-
_output_shapes
:€€€€€€€€€:
а
einput_layer/sparse_input_layer/input_layer/C23_embedding/C23_embedding_weights/SparseReshape/IdentityIdentity?input_layer/sparse_input_layer/input_layer/C23_embedding/lookup*#
_output_shapes
:€€€€€€€€€*
T0	
Я
]input_layer/sparse_input_layer/input_layer/C23_embedding/C23_embedding_weights/GreaterEqual/yConst*
value	B	 R *
_output_shapes
: *
dtype0	
я
[input_layer/sparse_input_layer/input_layer/C23_embedding/C23_embedding_weights/GreaterEqualGreaterEqualeinput_layer/sparse_input_layer/input_layer/C23_embedding/C23_embedding_weights/SparseReshape/Identity]input_layer/sparse_input_layer/input_layer/C23_embedding/C23_embedding_weights/GreaterEqual/y*
T0	*#
_output_shapes
:€€€€€€€€€
г
Tinput_layer/sparse_input_layer/input_layer/C23_embedding/C23_embedding_weights/WhereWhere[input_layer/sparse_input_layer/input_layer/C23_embedding/C23_embedding_weights/GreaterEqual*'
_output_shapes
:€€€€€€€€€
ѓ
\input_layer/sparse_input_layer/input_layer/C23_embedding/C23_embedding_weights/Reshape/shapeConst*
valueB:
€€€€€€€€€*
_output_shapes
:*
dtype0
√
Vinput_layer/sparse_input_layer/input_layer/C23_embedding/C23_embedding_weights/ReshapeReshapeTinput_layer/sparse_input_layer/input_layer/C23_embedding/C23_embedding_weights/Where\input_layer/sparse_input_layer/input_layer/C23_embedding/C23_embedding_weights/Reshape/shape*
T0	*#
_output_shapes
:€€€€€€€€€
†
^input_layer/sparse_input_layer/input_layer/C23_embedding/C23_embedding_weights/GatherV2_1/axisConst*
dtype0*
value	B : *
_output_shapes
: 
–
Yinput_layer/sparse_input_layer/input_layer/C23_embedding/C23_embedding_weights/GatherV2_1GatherV2\input_layer/sparse_input_layer/input_layer/C23_embedding/C23_embedding_weights/SparseReshapeVinput_layer/sparse_input_layer/input_layer/C23_embedding/C23_embedding_weights/Reshape^input_layer/sparse_input_layer/input_layer/C23_embedding/C23_embedding_weights/GatherV2_1/axis*
Tparams0	*'
_output_shapes
:€€€€€€€€€*
Taxis0*
Tindices0	
†
^input_layer/sparse_input_layer/input_layer/C23_embedding/C23_embedding_weights/GatherV2_2/axisConst*
_output_shapes
: *
value	B : *
dtype0
’
Yinput_layer/sparse_input_layer/input_layer/C23_embedding/C23_embedding_weights/GatherV2_2GatherV2einput_layer/sparse_input_layer/input_layer/C23_embedding/C23_embedding_weights/SparseReshape/IdentityVinput_layer/sparse_input_layer/input_layer/C23_embedding/C23_embedding_weights/Reshape^input_layer/sparse_input_layer/input_layer/C23_embedding/C23_embedding_weights/GatherV2_2/axis*
Taxis0*
Tparams0	*#
_output_shapes
:€€€€€€€€€*
Tindices0	
и
Winput_layer/sparse_input_layer/input_layer/C23_embedding/C23_embedding_weights/IdentityIdentity^input_layer/sparse_input_layer/input_layer/C23_embedding/C23_embedding_weights/SparseReshape:1*
_output_shapes
:*
T0	
™
hinput_layer/sparse_input_layer/input_layer/C23_embedding/C23_embedding_weights/SparseFillEmptyRows/ConstConst*
value	B	 R *
dtype0	*
_output_shapes
: 
е
vinput_layer/sparse_input_layer/input_layer/C23_embedding/C23_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowsSparseFillEmptyRowsYinput_layer/sparse_input_layer/input_layer/C23_embedding/C23_embedding_weights/GatherV2_1Yinput_layer/sparse_input_layer/input_layer/C23_embedding/C23_embedding_weights/GatherV2_2Winput_layer/sparse_input_layer/input_layer/C23_embedding/C23_embedding_weights/Identityhinput_layer/sparse_input_layer/input_layer/C23_embedding/C23_embedding_weights/SparseFillEmptyRows/Const*T
_output_shapesB
@:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€*
T0	
Ћ
zinput_layer/sparse_input_layer/input_layer/C23_embedding/C23_embedding_weights/embedding_lookup_sparse/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB"        
Ќ
|input_layer/sparse_input_layer/input_layer/C23_embedding/C23_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1Const*
valueB"       *
_output_shapes
:*
dtype0
Ќ
|input_layer/sparse_input_layer/input_layer/C23_embedding/C23_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2Const*
dtype0*
valueB"      *
_output_shapes
:
й
tinput_layer/sparse_input_layer/input_layer/C23_embedding/C23_embedding_weights/embedding_lookup_sparse/strided_sliceStridedSlicevinput_layer/sparse_input_layer/input_layer/C23_embedding/C23_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowszinput_layer/sparse_input_layer/input_layer/C23_embedding/C23_embedding_weights/embedding_lookup_sparse/strided_slice/stack|input_layer/sparse_input_layer/input_layer/C23_embedding/C23_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1|input_layer/sparse_input_layer/input_layer/C23_embedding/C23_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2*
T0	*#
_output_shapes
:€€€€€€€€€*
shrink_axis_mask*

begin_mask*
Index0*
end_mask
¶
kinput_layer/sparse_input_layer/input_layer/C23_embedding/C23_embedding_weights/embedding_lookup_sparse/CastCasttinput_layer/sparse_input_layer/input_layer/C23_embedding/C23_embedding_weights/embedding_lookup_sparse/strided_slice*#
_output_shapes
:€€€€€€€€€*

DstT0*

SrcT0	
Ѓ
minput_layer/sparse_input_layer/input_layer/C23_embedding/C23_embedding_weights/embedding_lookup_sparse/UniqueUniquexinput_layer/sparse_input_layer/input_layer/C23_embedding/C23_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:1*
T0	*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
Э
|input_layer/sparse_input_layer/input_layer/C23_embedding/C23_embedding_weights/embedding_lookup_sparse/embedding_lookup/axisConst*]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C23_embedding/embedding_weights*
value	B : *
dtype0*
_output_shapes
: 
х
winput_layer/sparse_input_layer/input_layer/C23_embedding/C23_embedding_weights/embedding_lookup_sparse/embedding_lookupGatherV2Oinput_layer/sparse_input_layer/input_layer/C23_embedding/embedding_weights/readminput_layer/sparse_input_layer/input_layer/C23_embedding/C23_embedding_weights/embedding_lookup_sparse/Unique|input_layer/sparse_input_layer/input_layer/C23_embedding/C23_embedding_weights/embedding_lookup_sparse/embedding_lookup/axis*
Tindices0	*]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C23_embedding/embedding_weights*'
_output_shapes
:€€€€€€€€€*
Taxis0*
Tparams0
Є
Аinput_layer/sparse_input_layer/input_layer/C23_embedding/C23_embedding_weights/embedding_lookup_sparse/embedding_lookup/IdentityIdentitywinput_layer/sparse_input_layer/input_layer/C23_embedding/C23_embedding_weights/embedding_lookup_sparse/embedding_lookup*
T0*'
_output_shapes
:€€€€€€€€€
О
finput_layer/sparse_input_layer/input_layer/C23_embedding/C23_embedding_weights/embedding_lookup_sparseSparseSegmentMeanАinput_layer/sparse_input_layer/input_layer/C23_embedding/C23_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identityoinput_layer/sparse_input_layer/input_layer/C23_embedding/C23_embedding_weights/embedding_lookup_sparse/Unique:1kinput_layer/sparse_input_layer/input_layer/C23_embedding/C23_embedding_weights/embedding_lookup_sparse/Cast*'
_output_shapes
:€€€€€€€€€*
T0
ѓ
^input_layer/sparse_input_layer/input_layer/C23_embedding/C23_embedding_weights/Reshape_1/shapeConst*
valueB"€€€€   *
dtype0*
_output_shapes
:
п
Xinput_layer/sparse_input_layer/input_layer/C23_embedding/C23_embedding_weights/Reshape_1Reshapexinput_layer/sparse_input_layer/input_layer/C23_embedding/C23_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:2^input_layer/sparse_input_layer/input_layer/C23_embedding/C23_embedding_weights/Reshape_1/shape*'
_output_shapes
:€€€€€€€€€*
T0

к
Tinput_layer/sparse_input_layer/input_layer/C23_embedding/C23_embedding_weights/ShapeShapefinput_layer/sparse_input_layer/input_layer/C23_embedding/C23_embedding_weights/embedding_lookup_sparse*
T0*
_output_shapes
:
ђ
binput_layer/sparse_input_layer/input_layer/C23_embedding/C23_embedding_weights/strided_slice/stackConst*
_output_shapes
:*
valueB:*
dtype0
Ѓ
dinput_layer/sparse_input_layer/input_layer/C23_embedding/C23_embedding_weights/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
Ѓ
dinput_layer/sparse_input_layer/input_layer/C23_embedding/C23_embedding_weights/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Є
\input_layer/sparse_input_layer/input_layer/C23_embedding/C23_embedding_weights/strided_sliceStridedSliceTinput_layer/sparse_input_layer/input_layer/C23_embedding/C23_embedding_weights/Shapebinput_layer/sparse_input_layer/input_layer/C23_embedding/C23_embedding_weights/strided_slice/stackdinput_layer/sparse_input_layer/input_layer/C23_embedding/C23_embedding_weights/strided_slice/stack_1dinput_layer/sparse_input_layer/input_layer/C23_embedding/C23_embedding_weights/strided_slice/stack_2*
_output_shapes
: *
Index0*
T0*
shrink_axis_mask
Ш
Vinput_layer/sparse_input_layer/input_layer/C23_embedding/C23_embedding_weights/stack/0Const*
_output_shapes
: *
value	B :*
dtype0
ј
Tinput_layer/sparse_input_layer/input_layer/C23_embedding/C23_embedding_weights/stackPackVinput_layer/sparse_input_layer/input_layer/C23_embedding/C23_embedding_weights/stack/0\input_layer/sparse_input_layer/input_layer/C23_embedding/C23_embedding_weights/strided_slice*
_output_shapes
:*
T0*
N
∆
Sinput_layer/sparse_input_layer/input_layer/C23_embedding/C23_embedding_weights/TileTileXinput_layer/sparse_input_layer/input_layer/C23_embedding/C23_embedding_weights/Reshape_1Tinput_layer/sparse_input_layer/input_layer/C23_embedding/C23_embedding_weights/stack*
T0
*0
_output_shapes
:€€€€€€€€€€€€€€€€€€
А
Yinput_layer/sparse_input_layer/input_layer/C23_embedding/C23_embedding_weights/zeros_like	ZerosLikefinput_layer/sparse_input_layer/input_layer/C23_embedding/C23_embedding_weights/embedding_lookup_sparse*'
_output_shapes
:€€€€€€€€€*
T0
Ґ
Ninput_layer/sparse_input_layer/input_layer/C23_embedding/C23_embedding_weightsSelectSinput_layer/sparse_input_layer/input_layer/C23_embedding/C23_embedding_weights/TileYinput_layer/sparse_input_layer/input_layer/C23_embedding/C23_embedding_weights/zeros_likefinput_layer/sparse_input_layer/input_layer/C23_embedding/C23_embedding_weights/embedding_lookup_sparse*
T0*'
_output_shapes
:€€€€€€€€€
з
Uinput_layer/sparse_input_layer/input_layer/C23_embedding/C23_embedding_weights/Cast_1CastTinput_layer/sparse_input_layer/input_layer/C23_embedding/to_sparse_input/dense_shape*

DstT0*
_output_shapes
:*

SrcT0	
¶
\input_layer/sparse_input_layer/input_layer/C23_embedding/C23_embedding_weights/Slice_1/beginConst*
valueB: *
_output_shapes
:*
dtype0
•
[input_layer/sparse_input_layer/input_layer/C23_embedding/C23_embedding_weights/Slice_1/sizeConst*
_output_shapes
:*
dtype0*
valueB:
£
Vinput_layer/sparse_input_layer/input_layer/C23_embedding/C23_embedding_weights/Slice_1SliceUinput_layer/sparse_input_layer/input_layer/C23_embedding/C23_embedding_weights/Cast_1\input_layer/sparse_input_layer/input_layer/C23_embedding/C23_embedding_weights/Slice_1/begin[input_layer/sparse_input_layer/input_layer/C23_embedding/C23_embedding_weights/Slice_1/size*
Index0*
T0*
_output_shapes
:
‘
Vinput_layer/sparse_input_layer/input_layer/C23_embedding/C23_embedding_weights/Shape_1ShapeNinput_layer/sparse_input_layer/input_layer/C23_embedding/C23_embedding_weights*
T0*
_output_shapes
:
¶
\input_layer/sparse_input_layer/input_layer/C23_embedding/C23_embedding_weights/Slice_2/beginConst*
_output_shapes
:*
valueB:*
dtype0
Ѓ
[input_layer/sparse_input_layer/input_layer/C23_embedding/C23_embedding_weights/Slice_2/sizeConst*
dtype0*
valueB:
€€€€€€€€€*
_output_shapes
:
§
Vinput_layer/sparse_input_layer/input_layer/C23_embedding/C23_embedding_weights/Slice_2SliceVinput_layer/sparse_input_layer/input_layer/C23_embedding/C23_embedding_weights/Shape_1\input_layer/sparse_input_layer/input_layer/C23_embedding/C23_embedding_weights/Slice_2/begin[input_layer/sparse_input_layer/input_layer/C23_embedding/C23_embedding_weights/Slice_2/size*
T0*
Index0*
_output_shapes
:
Ь
Zinput_layer/sparse_input_layer/input_layer/C23_embedding/C23_embedding_weights/concat/axisConst*
value	B : *
_output_shapes
: *
dtype0
Ы
Uinput_layer/sparse_input_layer/input_layer/C23_embedding/C23_embedding_weights/concatConcatV2Vinput_layer/sparse_input_layer/input_layer/C23_embedding/C23_embedding_weights/Slice_1Vinput_layer/sparse_input_layer/input_layer/C23_embedding/C23_embedding_weights/Slice_2Zinput_layer/sparse_input_layer/input_layer/C23_embedding/C23_embedding_weights/concat/axis*
T0*
N*
_output_shapes
:
Љ
Xinput_layer/sparse_input_layer/input_layer/C23_embedding/C23_embedding_weights/Reshape_2ReshapeNinput_layer/sparse_input_layer/input_layer/C23_embedding/C23_embedding_weightsUinput_layer/sparse_input_layer/input_layer/C23_embedding/C23_embedding_weights/concat*
T0*'
_output_shapes
:€€€€€€€€€
∆
>input_layer/sparse_input_layer/input_layer/C23_embedding/ShapeShapeXinput_layer/sparse_input_layer/input_layer/C23_embedding/C23_embedding_weights/Reshape_2*
T0*
_output_shapes
:
Ц
Linput_layer/sparse_input_layer/input_layer/C23_embedding/strided_slice/stackConst*
dtype0*
valueB: *
_output_shapes
:
Ш
Ninput_layer/sparse_input_layer/input_layer/C23_embedding/strided_slice/stack_1Const*
dtype0*
valueB:*
_output_shapes
:
Ш
Ninput_layer/sparse_input_layer/input_layer/C23_embedding/strided_slice/stack_2Const*
_output_shapes
:*
valueB:*
dtype0
 
Finput_layer/sparse_input_layer/input_layer/C23_embedding/strided_sliceStridedSlice>input_layer/sparse_input_layer/input_layer/C23_embedding/ShapeLinput_layer/sparse_input_layer/input_layer/C23_embedding/strided_slice/stackNinput_layer/sparse_input_layer/input_layer/C23_embedding/strided_slice/stack_1Ninput_layer/sparse_input_layer/input_layer/C23_embedding/strided_slice/stack_2*
Index0*
_output_shapes
: *
shrink_axis_mask*
T0
К
Hinput_layer/sparse_input_layer/input_layer/C23_embedding/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
О
Finput_layer/sparse_input_layer/input_layer/C23_embedding/Reshape/shapePackFinput_layer/sparse_input_layer/input_layer/C23_embedding/strided_sliceHinput_layer/sparse_input_layer/input_layer/C23_embedding/Reshape/shape/1*
T0*
_output_shapes
:*
N
Я
@input_layer/sparse_input_layer/input_layer/C23_embedding/ReshapeReshapeXinput_layer/sparse_input_layer/input_layer/C23_embedding/C23_embedding_weights/Reshape_2Finput_layer/sparse_input_layer/input_layer/C23_embedding/Reshape/shape*
T0*'
_output_shapes
:€€€€€€€€€
Т
Ginput_layer/sparse_input_layer/input_layer/C24_embedding/ExpandDims/dimConst*
valueB :
€€€€€€€€€*
_output_shapes
: *
dtype0
№
Cinput_layer/sparse_input_layer/input_layer/C24_embedding/ExpandDims
ExpandDimsPlaceholder_36Ginput_layer/sparse_input_layer/input_layer/C24_embedding/ExpandDims/dim*
T0*'
_output_shapes
:€€€€€€€€€
Ш
Winput_layer/sparse_input_layer/input_layer/C24_embedding/to_sparse_input/ignore_value/xConst*
dtype0*
valueB B *
_output_shapes
: 
≠
Qinput_layer/sparse_input_layer/input_layer/C24_embedding/to_sparse_input/NotEqualNotEqualCinput_layer/sparse_input_layer/input_layer/C24_embedding/ExpandDimsWinput_layer/sparse_input_layer/input_layer/C24_embedding/to_sparse_input/ignore_value/x*
T0*'
_output_shapes
:€€€€€€€€€
’
Pinput_layer/sparse_input_layer/input_layer/C24_embedding/to_sparse_input/indicesWhereQinput_layer/sparse_input_layer/input_layer/C24_embedding/to_sparse_input/NotEqual*'
_output_shapes
:€€€€€€€€€
ґ
Oinput_layer/sparse_input_layer/input_layer/C24_embedding/to_sparse_input/valuesGatherNdCinput_layer/sparse_input_layer/input_layer/C24_embedding/ExpandDimsPinput_layer/sparse_input_layer/input_layer/C24_embedding/to_sparse_input/indices*
Tindices0	*
Tparams0*#
_output_shapes
:€€€€€€€€€
„
Tinput_layer/sparse_input_layer/input_layer/C24_embedding/to_sparse_input/dense_shapeShapeCinput_layer/sparse_input_layer/input_layer/C24_embedding/ExpandDims*
out_type0	*
T0*
_output_shapes
:
г
?input_layer/sparse_input_layer/input_layer/C24_embedding/lookupStringToHashBucketFastOinput_layer/sparse_input_layer/input_layer/C24_embedding/to_sparse_input/values*
num_bucketsРN*#
_output_shapes
:€€€€€€€€€
Э
minput_layer/sparse_input_layer/input_layer/C24_embedding/embedding_weights/Initializer/truncated_normal/shapeConst*
dtype0*
_output_shapes
:*]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C24_embedding/embedding_weights*
valueB"'     
Р
linput_layer/sparse_input_layer/input_layer/C24_embedding/embedding_weights/Initializer/truncated_normal/meanConst*
dtype0*]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C24_embedding/embedding_weights*
valueB
 *    *
_output_shapes
: 
Т
ninput_layer/sparse_input_layer/input_layer/C24_embedding/embedding_weights/Initializer/truncated_normal/stddevConst*
valueB
 *  А>*]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C24_embedding/embedding_weights*
_output_shapes
: *
dtype0
П
winput_layer/sparse_input_layer/input_layer/C24_embedding/embedding_weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormalminput_layer/sparse_input_layer/input_layer/C24_embedding/embedding_weights/Initializer/truncated_normal/shape*]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C24_embedding/embedding_weights*
dtype0*
T0*
_output_shapes
:	РN
д
kinput_layer/sparse_input_layer/input_layer/C24_embedding/embedding_weights/Initializer/truncated_normal/mulMulwinput_layer/sparse_input_layer/input_layer/C24_embedding/embedding_weights/Initializer/truncated_normal/TruncatedNormalninput_layer/sparse_input_layer/input_layer/C24_embedding/embedding_weights/Initializer/truncated_normal/stddev*
T0*
_output_shapes
:	РN*]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C24_embedding/embedding_weights
“
ginput_layer/sparse_input_layer/input_layer/C24_embedding/embedding_weights/Initializer/truncated_normalAddkinput_layer/sparse_input_layer/input_layer/C24_embedding/embedding_weights/Initializer/truncated_normal/mullinput_layer/sparse_input_layer/input_layer/C24_embedding/embedding_weights/Initializer/truncated_normal/mean*]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C24_embedding/embedding_weights*
T0*
_output_shapes
:	РN
ы
Jinput_layer/sparse_input_layer/input_layer/C24_embedding/embedding_weights
VariableV2*
_output_shapes
:	РN*]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C24_embedding/embedding_weights*
shape:	РN*
dtype0
Щ
Qinput_layer/sparse_input_layer/input_layer/C24_embedding/embedding_weights/AssignAssignJinput_layer/sparse_input_layer/input_layer/C24_embedding/embedding_weightsginput_layer/sparse_input_layer/input_layer/C24_embedding/embedding_weights/Initializer/truncated_normal*]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C24_embedding/embedding_weights*
T0*
_output_shapes
:	РN
∞
Oinput_layer/sparse_input_layer/input_layer/C24_embedding/embedding_weights/readIdentityJinput_layer/sparse_input_layer/input_layer/C24_embedding/embedding_weights*
T0*
_output_shapes
:	РN*]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C24_embedding/embedding_weights
§
Zinput_layer/sparse_input_layer/input_layer/C24_embedding/C24_embedding_weights/Slice/beginConst*
valueB: *
dtype0*
_output_shapes
:
£
Yinput_layer/sparse_input_layer/input_layer/C24_embedding/C24_embedding_weights/Slice/sizeConst*
_output_shapes
:*
dtype0*
valueB:
Ь
Tinput_layer/sparse_input_layer/input_layer/C24_embedding/C24_embedding_weights/SliceSliceTinput_layer/sparse_input_layer/input_layer/C24_embedding/to_sparse_input/dense_shapeZinput_layer/sparse_input_layer/input_layer/C24_embedding/C24_embedding_weights/Slice/beginYinput_layer/sparse_input_layer/input_layer/C24_embedding/C24_embedding_weights/Slice/size*
_output_shapes
:*
Index0*
T0	
Ю
Tinput_layer/sparse_input_layer/input_layer/C24_embedding/C24_embedding_weights/ConstConst*
valueB: *
dtype0*
_output_shapes
:
®
Sinput_layer/sparse_input_layer/input_layer/C24_embedding/C24_embedding_weights/ProdProdTinput_layer/sparse_input_layer/input_layer/C24_embedding/C24_embedding_weights/SliceTinput_layer/sparse_input_layer/input_layer/C24_embedding/C24_embedding_weights/Const*
_output_shapes
: *
T0	
°
_input_layer/sparse_input_layer/input_layer/C24_embedding/C24_embedding_weights/GatherV2/indicesConst*
value	B :*
dtype0*
_output_shapes
: 
Ю
\input_layer/sparse_input_layer/input_layer/C24_embedding/C24_embedding_weights/GatherV2/axisConst*
dtype0*
value	B : *
_output_shapes
: 
Љ
Winput_layer/sparse_input_layer/input_layer/C24_embedding/C24_embedding_weights/GatherV2GatherV2Tinput_layer/sparse_input_layer/input_layer/C24_embedding/to_sparse_input/dense_shape_input_layer/sparse_input_layer/input_layer/C24_embedding/C24_embedding_weights/GatherV2/indices\input_layer/sparse_input_layer/input_layer/C24_embedding/C24_embedding_weights/GatherV2/axis*
Tparams0	*
Taxis0*
_output_shapes
: *
Tindices0
є
Uinput_layer/sparse_input_layer/input_layer/C24_embedding/C24_embedding_weights/Cast/xPackSinput_layer/sparse_input_layer/input_layer/C24_embedding/C24_embedding_weights/ProdWinput_layer/sparse_input_layer/input_layer/C24_embedding/C24_embedding_weights/GatherV2*
_output_shapes
:*
N*
T0	
Ы
\input_layer/sparse_input_layer/input_layer/C24_embedding/C24_embedding_weights/SparseReshapeSparseReshapePinput_layer/sparse_input_layer/input_layer/C24_embedding/to_sparse_input/indicesTinput_layer/sparse_input_layer/input_layer/C24_embedding/to_sparse_input/dense_shapeUinput_layer/sparse_input_layer/input_layer/C24_embedding/C24_embedding_weights/Cast/x*-
_output_shapes
:€€€€€€€€€:
а
einput_layer/sparse_input_layer/input_layer/C24_embedding/C24_embedding_weights/SparseReshape/IdentityIdentity?input_layer/sparse_input_layer/input_layer/C24_embedding/lookup*#
_output_shapes
:€€€€€€€€€*
T0	
Я
]input_layer/sparse_input_layer/input_layer/C24_embedding/C24_embedding_weights/GreaterEqual/yConst*
_output_shapes
: *
value	B	 R *
dtype0	
я
[input_layer/sparse_input_layer/input_layer/C24_embedding/C24_embedding_weights/GreaterEqualGreaterEqualeinput_layer/sparse_input_layer/input_layer/C24_embedding/C24_embedding_weights/SparseReshape/Identity]input_layer/sparse_input_layer/input_layer/C24_embedding/C24_embedding_weights/GreaterEqual/y*#
_output_shapes
:€€€€€€€€€*
T0	
г
Tinput_layer/sparse_input_layer/input_layer/C24_embedding/C24_embedding_weights/WhereWhere[input_layer/sparse_input_layer/input_layer/C24_embedding/C24_embedding_weights/GreaterEqual*'
_output_shapes
:€€€€€€€€€
ѓ
\input_layer/sparse_input_layer/input_layer/C24_embedding/C24_embedding_weights/Reshape/shapeConst*
dtype0*
valueB:
€€€€€€€€€*
_output_shapes
:
√
Vinput_layer/sparse_input_layer/input_layer/C24_embedding/C24_embedding_weights/ReshapeReshapeTinput_layer/sparse_input_layer/input_layer/C24_embedding/C24_embedding_weights/Where\input_layer/sparse_input_layer/input_layer/C24_embedding/C24_embedding_weights/Reshape/shape*
T0	*#
_output_shapes
:€€€€€€€€€
†
^input_layer/sparse_input_layer/input_layer/C24_embedding/C24_embedding_weights/GatherV2_1/axisConst*
dtype0*
_output_shapes
: *
value	B : 
–
Yinput_layer/sparse_input_layer/input_layer/C24_embedding/C24_embedding_weights/GatherV2_1GatherV2\input_layer/sparse_input_layer/input_layer/C24_embedding/C24_embedding_weights/SparseReshapeVinput_layer/sparse_input_layer/input_layer/C24_embedding/C24_embedding_weights/Reshape^input_layer/sparse_input_layer/input_layer/C24_embedding/C24_embedding_weights/GatherV2_1/axis*'
_output_shapes
:€€€€€€€€€*
Tindices0	*
Taxis0*
Tparams0	
†
^input_layer/sparse_input_layer/input_layer/C24_embedding/C24_embedding_weights/GatherV2_2/axisConst*
_output_shapes
: *
value	B : *
dtype0
’
Yinput_layer/sparse_input_layer/input_layer/C24_embedding/C24_embedding_weights/GatherV2_2GatherV2einput_layer/sparse_input_layer/input_layer/C24_embedding/C24_embedding_weights/SparseReshape/IdentityVinput_layer/sparse_input_layer/input_layer/C24_embedding/C24_embedding_weights/Reshape^input_layer/sparse_input_layer/input_layer/C24_embedding/C24_embedding_weights/GatherV2_2/axis*
Taxis0*#
_output_shapes
:€€€€€€€€€*
Tindices0	*
Tparams0	
и
Winput_layer/sparse_input_layer/input_layer/C24_embedding/C24_embedding_weights/IdentityIdentity^input_layer/sparse_input_layer/input_layer/C24_embedding/C24_embedding_weights/SparseReshape:1*
T0	*
_output_shapes
:
™
hinput_layer/sparse_input_layer/input_layer/C24_embedding/C24_embedding_weights/SparseFillEmptyRows/ConstConst*
dtype0	*
_output_shapes
: *
value	B	 R 
е
vinput_layer/sparse_input_layer/input_layer/C24_embedding/C24_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowsSparseFillEmptyRowsYinput_layer/sparse_input_layer/input_layer/C24_embedding/C24_embedding_weights/GatherV2_1Yinput_layer/sparse_input_layer/input_layer/C24_embedding/C24_embedding_weights/GatherV2_2Winput_layer/sparse_input_layer/input_layer/C24_embedding/C24_embedding_weights/Identityhinput_layer/sparse_input_layer/input_layer/C24_embedding/C24_embedding_weights/SparseFillEmptyRows/Const*
T0	*T
_output_shapesB
@:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€
Ћ
zinput_layer/sparse_input_layer/input_layer/C24_embedding/C24_embedding_weights/embedding_lookup_sparse/strided_slice/stackConst*
dtype0*
valueB"        *
_output_shapes
:
Ќ
|input_layer/sparse_input_layer/input_layer/C24_embedding/C24_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       
Ќ
|input_layer/sparse_input_layer/input_layer/C24_embedding/C24_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
й
tinput_layer/sparse_input_layer/input_layer/C24_embedding/C24_embedding_weights/embedding_lookup_sparse/strided_sliceStridedSlicevinput_layer/sparse_input_layer/input_layer/C24_embedding/C24_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowszinput_layer/sparse_input_layer/input_layer/C24_embedding/C24_embedding_weights/embedding_lookup_sparse/strided_slice/stack|input_layer/sparse_input_layer/input_layer/C24_embedding/C24_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1|input_layer/sparse_input_layer/input_layer/C24_embedding/C24_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2*#
_output_shapes
:€€€€€€€€€*
end_mask*
shrink_axis_mask*

begin_mask*
Index0*
T0	
¶
kinput_layer/sparse_input_layer/input_layer/C24_embedding/C24_embedding_weights/embedding_lookup_sparse/CastCasttinput_layer/sparse_input_layer/input_layer/C24_embedding/C24_embedding_weights/embedding_lookup_sparse/strided_slice*

SrcT0	*

DstT0*#
_output_shapes
:€€€€€€€€€
Ѓ
minput_layer/sparse_input_layer/input_layer/C24_embedding/C24_embedding_weights/embedding_lookup_sparse/UniqueUniquexinput_layer/sparse_input_layer/input_layer/C24_embedding/C24_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:1*
T0	*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
Э
|input_layer/sparse_input_layer/input_layer/C24_embedding/C24_embedding_weights/embedding_lookup_sparse/embedding_lookup/axisConst*]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C24_embedding/embedding_weights*
dtype0*
value	B : *
_output_shapes
: 
х
winput_layer/sparse_input_layer/input_layer/C24_embedding/C24_embedding_weights/embedding_lookup_sparse/embedding_lookupGatherV2Oinput_layer/sparse_input_layer/input_layer/C24_embedding/embedding_weights/readminput_layer/sparse_input_layer/input_layer/C24_embedding/C24_embedding_weights/embedding_lookup_sparse/Unique|input_layer/sparse_input_layer/input_layer/C24_embedding/C24_embedding_weights/embedding_lookup_sparse/embedding_lookup/axis*
Tparams0*'
_output_shapes
:€€€€€€€€€*
Taxis0*]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C24_embedding/embedding_weights*
Tindices0	
Є
Аinput_layer/sparse_input_layer/input_layer/C24_embedding/C24_embedding_weights/embedding_lookup_sparse/embedding_lookup/IdentityIdentitywinput_layer/sparse_input_layer/input_layer/C24_embedding/C24_embedding_weights/embedding_lookup_sparse/embedding_lookup*
T0*'
_output_shapes
:€€€€€€€€€
О
finput_layer/sparse_input_layer/input_layer/C24_embedding/C24_embedding_weights/embedding_lookup_sparseSparseSegmentMeanАinput_layer/sparse_input_layer/input_layer/C24_embedding/C24_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identityoinput_layer/sparse_input_layer/input_layer/C24_embedding/C24_embedding_weights/embedding_lookup_sparse/Unique:1kinput_layer/sparse_input_layer/input_layer/C24_embedding/C24_embedding_weights/embedding_lookup_sparse/Cast*
T0*'
_output_shapes
:€€€€€€€€€
ѓ
^input_layer/sparse_input_layer/input_layer/C24_embedding/C24_embedding_weights/Reshape_1/shapeConst*
dtype0*
_output_shapes
:*
valueB"€€€€   
п
Xinput_layer/sparse_input_layer/input_layer/C24_embedding/C24_embedding_weights/Reshape_1Reshapexinput_layer/sparse_input_layer/input_layer/C24_embedding/C24_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:2^input_layer/sparse_input_layer/input_layer/C24_embedding/C24_embedding_weights/Reshape_1/shape*
T0
*'
_output_shapes
:€€€€€€€€€
к
Tinput_layer/sparse_input_layer/input_layer/C24_embedding/C24_embedding_weights/ShapeShapefinput_layer/sparse_input_layer/input_layer/C24_embedding/C24_embedding_weights/embedding_lookup_sparse*
T0*
_output_shapes
:
ђ
binput_layer/sparse_input_layer/input_layer/C24_embedding/C24_embedding_weights/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
Ѓ
dinput_layer/sparse_input_layer/input_layer/C24_embedding/C24_embedding_weights/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
Ѓ
dinput_layer/sparse_input_layer/input_layer/C24_embedding/C24_embedding_weights/strided_slice/stack_2Const*
_output_shapes
:*
valueB:*
dtype0
Є
\input_layer/sparse_input_layer/input_layer/C24_embedding/C24_embedding_weights/strided_sliceStridedSliceTinput_layer/sparse_input_layer/input_layer/C24_embedding/C24_embedding_weights/Shapebinput_layer/sparse_input_layer/input_layer/C24_embedding/C24_embedding_weights/strided_slice/stackdinput_layer/sparse_input_layer/input_layer/C24_embedding/C24_embedding_weights/strided_slice/stack_1dinput_layer/sparse_input_layer/input_layer/C24_embedding/C24_embedding_weights/strided_slice/stack_2*
Index0*
_output_shapes
: *
shrink_axis_mask*
T0
Ш
Vinput_layer/sparse_input_layer/input_layer/C24_embedding/C24_embedding_weights/stack/0Const*
dtype0*
_output_shapes
: *
value	B :
ј
Tinput_layer/sparse_input_layer/input_layer/C24_embedding/C24_embedding_weights/stackPackVinput_layer/sparse_input_layer/input_layer/C24_embedding/C24_embedding_weights/stack/0\input_layer/sparse_input_layer/input_layer/C24_embedding/C24_embedding_weights/strided_slice*
T0*
N*
_output_shapes
:
∆
Sinput_layer/sparse_input_layer/input_layer/C24_embedding/C24_embedding_weights/TileTileXinput_layer/sparse_input_layer/input_layer/C24_embedding/C24_embedding_weights/Reshape_1Tinput_layer/sparse_input_layer/input_layer/C24_embedding/C24_embedding_weights/stack*0
_output_shapes
:€€€€€€€€€€€€€€€€€€*
T0

А
Yinput_layer/sparse_input_layer/input_layer/C24_embedding/C24_embedding_weights/zeros_like	ZerosLikefinput_layer/sparse_input_layer/input_layer/C24_embedding/C24_embedding_weights/embedding_lookup_sparse*
T0*'
_output_shapes
:€€€€€€€€€
Ґ
Ninput_layer/sparse_input_layer/input_layer/C24_embedding/C24_embedding_weightsSelectSinput_layer/sparse_input_layer/input_layer/C24_embedding/C24_embedding_weights/TileYinput_layer/sparse_input_layer/input_layer/C24_embedding/C24_embedding_weights/zeros_likefinput_layer/sparse_input_layer/input_layer/C24_embedding/C24_embedding_weights/embedding_lookup_sparse*'
_output_shapes
:€€€€€€€€€*
T0
з
Uinput_layer/sparse_input_layer/input_layer/C24_embedding/C24_embedding_weights/Cast_1CastTinput_layer/sparse_input_layer/input_layer/C24_embedding/to_sparse_input/dense_shape*

DstT0*
_output_shapes
:*

SrcT0	
¶
\input_layer/sparse_input_layer/input_layer/C24_embedding/C24_embedding_weights/Slice_1/beginConst*
dtype0*
valueB: *
_output_shapes
:
•
[input_layer/sparse_input_layer/input_layer/C24_embedding/C24_embedding_weights/Slice_1/sizeConst*
valueB:*
dtype0*
_output_shapes
:
£
Vinput_layer/sparse_input_layer/input_layer/C24_embedding/C24_embedding_weights/Slice_1SliceUinput_layer/sparse_input_layer/input_layer/C24_embedding/C24_embedding_weights/Cast_1\input_layer/sparse_input_layer/input_layer/C24_embedding/C24_embedding_weights/Slice_1/begin[input_layer/sparse_input_layer/input_layer/C24_embedding/C24_embedding_weights/Slice_1/size*
_output_shapes
:*
T0*
Index0
‘
Vinput_layer/sparse_input_layer/input_layer/C24_embedding/C24_embedding_weights/Shape_1ShapeNinput_layer/sparse_input_layer/input_layer/C24_embedding/C24_embedding_weights*
_output_shapes
:*
T0
¶
\input_layer/sparse_input_layer/input_layer/C24_embedding/C24_embedding_weights/Slice_2/beginConst*
_output_shapes
:*
valueB:*
dtype0
Ѓ
[input_layer/sparse_input_layer/input_layer/C24_embedding/C24_embedding_weights/Slice_2/sizeConst*
dtype0*
valueB:
€€€€€€€€€*
_output_shapes
:
§
Vinput_layer/sparse_input_layer/input_layer/C24_embedding/C24_embedding_weights/Slice_2SliceVinput_layer/sparse_input_layer/input_layer/C24_embedding/C24_embedding_weights/Shape_1\input_layer/sparse_input_layer/input_layer/C24_embedding/C24_embedding_weights/Slice_2/begin[input_layer/sparse_input_layer/input_layer/C24_embedding/C24_embedding_weights/Slice_2/size*
T0*
Index0*
_output_shapes
:
Ь
Zinput_layer/sparse_input_layer/input_layer/C24_embedding/C24_embedding_weights/concat/axisConst*
dtype0*
_output_shapes
: *
value	B : 
Ы
Uinput_layer/sparse_input_layer/input_layer/C24_embedding/C24_embedding_weights/concatConcatV2Vinput_layer/sparse_input_layer/input_layer/C24_embedding/C24_embedding_weights/Slice_1Vinput_layer/sparse_input_layer/input_layer/C24_embedding/C24_embedding_weights/Slice_2Zinput_layer/sparse_input_layer/input_layer/C24_embedding/C24_embedding_weights/concat/axis*
N*
T0*
_output_shapes
:
Љ
Xinput_layer/sparse_input_layer/input_layer/C24_embedding/C24_embedding_weights/Reshape_2ReshapeNinput_layer/sparse_input_layer/input_layer/C24_embedding/C24_embedding_weightsUinput_layer/sparse_input_layer/input_layer/C24_embedding/C24_embedding_weights/concat*
T0*'
_output_shapes
:€€€€€€€€€
∆
>input_layer/sparse_input_layer/input_layer/C24_embedding/ShapeShapeXinput_layer/sparse_input_layer/input_layer/C24_embedding/C24_embedding_weights/Reshape_2*
_output_shapes
:*
T0
Ц
Linput_layer/sparse_input_layer/input_layer/C24_embedding/strided_slice/stackConst*
dtype0*
valueB: *
_output_shapes
:
Ш
Ninput_layer/sparse_input_layer/input_layer/C24_embedding/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
Ш
Ninput_layer/sparse_input_layer/input_layer/C24_embedding/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
 
Finput_layer/sparse_input_layer/input_layer/C24_embedding/strided_sliceStridedSlice>input_layer/sparse_input_layer/input_layer/C24_embedding/ShapeLinput_layer/sparse_input_layer/input_layer/C24_embedding/strided_slice/stackNinput_layer/sparse_input_layer/input_layer/C24_embedding/strided_slice/stack_1Ninput_layer/sparse_input_layer/input_layer/C24_embedding/strided_slice/stack_2*
T0*
_output_shapes
: *
Index0*
shrink_axis_mask
К
Hinput_layer/sparse_input_layer/input_layer/C24_embedding/Reshape/shape/1Const*
_output_shapes
: *
value	B :*
dtype0
О
Finput_layer/sparse_input_layer/input_layer/C24_embedding/Reshape/shapePackFinput_layer/sparse_input_layer/input_layer/C24_embedding/strided_sliceHinput_layer/sparse_input_layer/input_layer/C24_embedding/Reshape/shape/1*
T0*
N*
_output_shapes
:
Я
@input_layer/sparse_input_layer/input_layer/C24_embedding/ReshapeReshapeXinput_layer/sparse_input_layer/input_layer/C24_embedding/C24_embedding_weights/Reshape_2Finput_layer/sparse_input_layer/input_layer/C24_embedding/Reshape/shape*'
_output_shapes
:€€€€€€€€€*
T0
Т
Ginput_layer/sparse_input_layer/input_layer/C25_embedding/ExpandDims/dimConst*
dtype0*
_output_shapes
: *
valueB :
€€€€€€€€€
№
Cinput_layer/sparse_input_layer/input_layer/C25_embedding/ExpandDims
ExpandDimsPlaceholder_37Ginput_layer/sparse_input_layer/input_layer/C25_embedding/ExpandDims/dim*'
_output_shapes
:€€€€€€€€€*
T0
Ш
Winput_layer/sparse_input_layer/input_layer/C25_embedding/to_sparse_input/ignore_value/xConst*
valueB B *
_output_shapes
: *
dtype0
≠
Qinput_layer/sparse_input_layer/input_layer/C25_embedding/to_sparse_input/NotEqualNotEqualCinput_layer/sparse_input_layer/input_layer/C25_embedding/ExpandDimsWinput_layer/sparse_input_layer/input_layer/C25_embedding/to_sparse_input/ignore_value/x*'
_output_shapes
:€€€€€€€€€*
T0
’
Pinput_layer/sparse_input_layer/input_layer/C25_embedding/to_sparse_input/indicesWhereQinput_layer/sparse_input_layer/input_layer/C25_embedding/to_sparse_input/NotEqual*'
_output_shapes
:€€€€€€€€€
ґ
Oinput_layer/sparse_input_layer/input_layer/C25_embedding/to_sparse_input/valuesGatherNdCinput_layer/sparse_input_layer/input_layer/C25_embedding/ExpandDimsPinput_layer/sparse_input_layer/input_layer/C25_embedding/to_sparse_input/indices*
Tparams0*#
_output_shapes
:€€€€€€€€€*
Tindices0	
„
Tinput_layer/sparse_input_layer/input_layer/C25_embedding/to_sparse_input/dense_shapeShapeCinput_layer/sparse_input_layer/input_layer/C25_embedding/ExpandDims*
T0*
_output_shapes
:*
out_type0	
г
?input_layer/sparse_input_layer/input_layer/C25_embedding/lookupStringToHashBucketFastOinput_layer/sparse_input_layer/input_layer/C25_embedding/to_sparse_input/values*
num_bucketsРN*#
_output_shapes
:€€€€€€€€€
Э
minput_layer/sparse_input_layer/input_layer/C25_embedding/embedding_weights/Initializer/truncated_normal/shapeConst*
dtype0*]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C25_embedding/embedding_weights*
valueB"'     *
_output_shapes
:
Р
linput_layer/sparse_input_layer/input_layer/C25_embedding/embedding_weights/Initializer/truncated_normal/meanConst*
dtype0*
valueB
 *    *
_output_shapes
: *]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C25_embedding/embedding_weights
Т
ninput_layer/sparse_input_layer/input_layer/C25_embedding/embedding_weights/Initializer/truncated_normal/stddevConst*
valueB
 *  А>*
_output_shapes
: *
dtype0*]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C25_embedding/embedding_weights
П
winput_layer/sparse_input_layer/input_layer/C25_embedding/embedding_weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormalminput_layer/sparse_input_layer/input_layer/C25_embedding/embedding_weights/Initializer/truncated_normal/shape*
dtype0*]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C25_embedding/embedding_weights*
_output_shapes
:	РN*
T0
д
kinput_layer/sparse_input_layer/input_layer/C25_embedding/embedding_weights/Initializer/truncated_normal/mulMulwinput_layer/sparse_input_layer/input_layer/C25_embedding/embedding_weights/Initializer/truncated_normal/TruncatedNormalninput_layer/sparse_input_layer/input_layer/C25_embedding/embedding_weights/Initializer/truncated_normal/stddev*
T0*]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C25_embedding/embedding_weights*
_output_shapes
:	РN
“
ginput_layer/sparse_input_layer/input_layer/C25_embedding/embedding_weights/Initializer/truncated_normalAddkinput_layer/sparse_input_layer/input_layer/C25_embedding/embedding_weights/Initializer/truncated_normal/mullinput_layer/sparse_input_layer/input_layer/C25_embedding/embedding_weights/Initializer/truncated_normal/mean*
_output_shapes
:	РN*
T0*]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C25_embedding/embedding_weights
ы
Jinput_layer/sparse_input_layer/input_layer/C25_embedding/embedding_weights
VariableV2*
dtype0*]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C25_embedding/embedding_weights*
_output_shapes
:	РN*
shape:	РN
Щ
Qinput_layer/sparse_input_layer/input_layer/C25_embedding/embedding_weights/AssignAssignJinput_layer/sparse_input_layer/input_layer/C25_embedding/embedding_weightsginput_layer/sparse_input_layer/input_layer/C25_embedding/embedding_weights/Initializer/truncated_normal*
_output_shapes
:	РN*
T0*]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C25_embedding/embedding_weights
∞
Oinput_layer/sparse_input_layer/input_layer/C25_embedding/embedding_weights/readIdentityJinput_layer/sparse_input_layer/input_layer/C25_embedding/embedding_weights*]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C25_embedding/embedding_weights*
_output_shapes
:	РN*
T0
§
Zinput_layer/sparse_input_layer/input_layer/C25_embedding/C25_embedding_weights/Slice/beginConst*
valueB: *
dtype0*
_output_shapes
:
£
Yinput_layer/sparse_input_layer/input_layer/C25_embedding/C25_embedding_weights/Slice/sizeConst*
_output_shapes
:*
dtype0*
valueB:
Ь
Tinput_layer/sparse_input_layer/input_layer/C25_embedding/C25_embedding_weights/SliceSliceTinput_layer/sparse_input_layer/input_layer/C25_embedding/to_sparse_input/dense_shapeZinput_layer/sparse_input_layer/input_layer/C25_embedding/C25_embedding_weights/Slice/beginYinput_layer/sparse_input_layer/input_layer/C25_embedding/C25_embedding_weights/Slice/size*
_output_shapes
:*
T0	*
Index0
Ю
Tinput_layer/sparse_input_layer/input_layer/C25_embedding/C25_embedding_weights/ConstConst*
dtype0*
valueB: *
_output_shapes
:
®
Sinput_layer/sparse_input_layer/input_layer/C25_embedding/C25_embedding_weights/ProdProdTinput_layer/sparse_input_layer/input_layer/C25_embedding/C25_embedding_weights/SliceTinput_layer/sparse_input_layer/input_layer/C25_embedding/C25_embedding_weights/Const*
T0	*
_output_shapes
: 
°
_input_layer/sparse_input_layer/input_layer/C25_embedding/C25_embedding_weights/GatherV2/indicesConst*
value	B :*
_output_shapes
: *
dtype0
Ю
\input_layer/sparse_input_layer/input_layer/C25_embedding/C25_embedding_weights/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 
Љ
Winput_layer/sparse_input_layer/input_layer/C25_embedding/C25_embedding_weights/GatherV2GatherV2Tinput_layer/sparse_input_layer/input_layer/C25_embedding/to_sparse_input/dense_shape_input_layer/sparse_input_layer/input_layer/C25_embedding/C25_embedding_weights/GatherV2/indices\input_layer/sparse_input_layer/input_layer/C25_embedding/C25_embedding_weights/GatherV2/axis*
Taxis0*
Tparams0	*
_output_shapes
: *
Tindices0
є
Uinput_layer/sparse_input_layer/input_layer/C25_embedding/C25_embedding_weights/Cast/xPackSinput_layer/sparse_input_layer/input_layer/C25_embedding/C25_embedding_weights/ProdWinput_layer/sparse_input_layer/input_layer/C25_embedding/C25_embedding_weights/GatherV2*
N*
_output_shapes
:*
T0	
Ы
\input_layer/sparse_input_layer/input_layer/C25_embedding/C25_embedding_weights/SparseReshapeSparseReshapePinput_layer/sparse_input_layer/input_layer/C25_embedding/to_sparse_input/indicesTinput_layer/sparse_input_layer/input_layer/C25_embedding/to_sparse_input/dense_shapeUinput_layer/sparse_input_layer/input_layer/C25_embedding/C25_embedding_weights/Cast/x*-
_output_shapes
:€€€€€€€€€:
а
einput_layer/sparse_input_layer/input_layer/C25_embedding/C25_embedding_weights/SparseReshape/IdentityIdentity?input_layer/sparse_input_layer/input_layer/C25_embedding/lookup*
T0	*#
_output_shapes
:€€€€€€€€€
Я
]input_layer/sparse_input_layer/input_layer/C25_embedding/C25_embedding_weights/GreaterEqual/yConst*
value	B	 R *
dtype0	*
_output_shapes
: 
я
[input_layer/sparse_input_layer/input_layer/C25_embedding/C25_embedding_weights/GreaterEqualGreaterEqualeinput_layer/sparse_input_layer/input_layer/C25_embedding/C25_embedding_weights/SparseReshape/Identity]input_layer/sparse_input_layer/input_layer/C25_embedding/C25_embedding_weights/GreaterEqual/y*#
_output_shapes
:€€€€€€€€€*
T0	
г
Tinput_layer/sparse_input_layer/input_layer/C25_embedding/C25_embedding_weights/WhereWhere[input_layer/sparse_input_layer/input_layer/C25_embedding/C25_embedding_weights/GreaterEqual*'
_output_shapes
:€€€€€€€€€
ѓ
\input_layer/sparse_input_layer/input_layer/C25_embedding/C25_embedding_weights/Reshape/shapeConst*
dtype0*
_output_shapes
:*
valueB:
€€€€€€€€€
√
Vinput_layer/sparse_input_layer/input_layer/C25_embedding/C25_embedding_weights/ReshapeReshapeTinput_layer/sparse_input_layer/input_layer/C25_embedding/C25_embedding_weights/Where\input_layer/sparse_input_layer/input_layer/C25_embedding/C25_embedding_weights/Reshape/shape*#
_output_shapes
:€€€€€€€€€*
T0	
†
^input_layer/sparse_input_layer/input_layer/C25_embedding/C25_embedding_weights/GatherV2_1/axisConst*
value	B : *
_output_shapes
: *
dtype0
–
Yinput_layer/sparse_input_layer/input_layer/C25_embedding/C25_embedding_weights/GatherV2_1GatherV2\input_layer/sparse_input_layer/input_layer/C25_embedding/C25_embedding_weights/SparseReshapeVinput_layer/sparse_input_layer/input_layer/C25_embedding/C25_embedding_weights/Reshape^input_layer/sparse_input_layer/input_layer/C25_embedding/C25_embedding_weights/GatherV2_1/axis*
Taxis0*
Tparams0	*'
_output_shapes
:€€€€€€€€€*
Tindices0	
†
^input_layer/sparse_input_layer/input_layer/C25_embedding/C25_embedding_weights/GatherV2_2/axisConst*
_output_shapes
: *
value	B : *
dtype0
’
Yinput_layer/sparse_input_layer/input_layer/C25_embedding/C25_embedding_weights/GatherV2_2GatherV2einput_layer/sparse_input_layer/input_layer/C25_embedding/C25_embedding_weights/SparseReshape/IdentityVinput_layer/sparse_input_layer/input_layer/C25_embedding/C25_embedding_weights/Reshape^input_layer/sparse_input_layer/input_layer/C25_embedding/C25_embedding_weights/GatherV2_2/axis*#
_output_shapes
:€€€€€€€€€*
Taxis0*
Tparams0	*
Tindices0	
и
Winput_layer/sparse_input_layer/input_layer/C25_embedding/C25_embedding_weights/IdentityIdentity^input_layer/sparse_input_layer/input_layer/C25_embedding/C25_embedding_weights/SparseReshape:1*
_output_shapes
:*
T0	
™
hinput_layer/sparse_input_layer/input_layer/C25_embedding/C25_embedding_weights/SparseFillEmptyRows/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R 
е
vinput_layer/sparse_input_layer/input_layer/C25_embedding/C25_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowsSparseFillEmptyRowsYinput_layer/sparse_input_layer/input_layer/C25_embedding/C25_embedding_weights/GatherV2_1Yinput_layer/sparse_input_layer/input_layer/C25_embedding/C25_embedding_weights/GatherV2_2Winput_layer/sparse_input_layer/input_layer/C25_embedding/C25_embedding_weights/Identityhinput_layer/sparse_input_layer/input_layer/C25_embedding/C25_embedding_weights/SparseFillEmptyRows/Const*
T0	*T
_output_shapesB
@:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€
Ћ
zinput_layer/sparse_input_layer/input_layer/C25_embedding/C25_embedding_weights/embedding_lookup_sparse/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        
Ќ
|input_layer/sparse_input_layer/input_layer/C25_embedding/C25_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1Const*
_output_shapes
:*
valueB"       *
dtype0
Ќ
|input_layer/sparse_input_layer/input_layer/C25_embedding/C25_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB"      
й
tinput_layer/sparse_input_layer/input_layer/C25_embedding/C25_embedding_weights/embedding_lookup_sparse/strided_sliceStridedSlicevinput_layer/sparse_input_layer/input_layer/C25_embedding/C25_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowszinput_layer/sparse_input_layer/input_layer/C25_embedding/C25_embedding_weights/embedding_lookup_sparse/strided_slice/stack|input_layer/sparse_input_layer/input_layer/C25_embedding/C25_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1|input_layer/sparse_input_layer/input_layer/C25_embedding/C25_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2*#
_output_shapes
:€€€€€€€€€*

begin_mask*
Index0*
end_mask*
T0	*
shrink_axis_mask
¶
kinput_layer/sparse_input_layer/input_layer/C25_embedding/C25_embedding_weights/embedding_lookup_sparse/CastCasttinput_layer/sparse_input_layer/input_layer/C25_embedding/C25_embedding_weights/embedding_lookup_sparse/strided_slice*

SrcT0	*#
_output_shapes
:€€€€€€€€€*

DstT0
Ѓ
minput_layer/sparse_input_layer/input_layer/C25_embedding/C25_embedding_weights/embedding_lookup_sparse/UniqueUniquexinput_layer/sparse_input_layer/input_layer/C25_embedding/C25_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€*
T0	
Э
|input_layer/sparse_input_layer/input_layer/C25_embedding/C25_embedding_weights/embedding_lookup_sparse/embedding_lookup/axisConst*]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C25_embedding/embedding_weights*
value	B : *
dtype0*
_output_shapes
: 
х
winput_layer/sparse_input_layer/input_layer/C25_embedding/C25_embedding_weights/embedding_lookup_sparse/embedding_lookupGatherV2Oinput_layer/sparse_input_layer/input_layer/C25_embedding/embedding_weights/readminput_layer/sparse_input_layer/input_layer/C25_embedding/C25_embedding_weights/embedding_lookup_sparse/Unique|input_layer/sparse_input_layer/input_layer/C25_embedding/C25_embedding_weights/embedding_lookup_sparse/embedding_lookup/axis*'
_output_shapes
:€€€€€€€€€*
Tindices0	*
Tparams0*
Taxis0*]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C25_embedding/embedding_weights
Є
Аinput_layer/sparse_input_layer/input_layer/C25_embedding/C25_embedding_weights/embedding_lookup_sparse/embedding_lookup/IdentityIdentitywinput_layer/sparse_input_layer/input_layer/C25_embedding/C25_embedding_weights/embedding_lookup_sparse/embedding_lookup*'
_output_shapes
:€€€€€€€€€*
T0
О
finput_layer/sparse_input_layer/input_layer/C25_embedding/C25_embedding_weights/embedding_lookup_sparseSparseSegmentMeanАinput_layer/sparse_input_layer/input_layer/C25_embedding/C25_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identityoinput_layer/sparse_input_layer/input_layer/C25_embedding/C25_embedding_weights/embedding_lookup_sparse/Unique:1kinput_layer/sparse_input_layer/input_layer/C25_embedding/C25_embedding_weights/embedding_lookup_sparse/Cast*
T0*'
_output_shapes
:€€€€€€€€€
ѓ
^input_layer/sparse_input_layer/input_layer/C25_embedding/C25_embedding_weights/Reshape_1/shapeConst*
valueB"€€€€   *
dtype0*
_output_shapes
:
п
Xinput_layer/sparse_input_layer/input_layer/C25_embedding/C25_embedding_weights/Reshape_1Reshapexinput_layer/sparse_input_layer/input_layer/C25_embedding/C25_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:2^input_layer/sparse_input_layer/input_layer/C25_embedding/C25_embedding_weights/Reshape_1/shape*
T0
*'
_output_shapes
:€€€€€€€€€
к
Tinput_layer/sparse_input_layer/input_layer/C25_embedding/C25_embedding_weights/ShapeShapefinput_layer/sparse_input_layer/input_layer/C25_embedding/C25_embedding_weights/embedding_lookup_sparse*
_output_shapes
:*
T0
ђ
binput_layer/sparse_input_layer/input_layer/C25_embedding/C25_embedding_weights/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
Ѓ
dinput_layer/sparse_input_layer/input_layer/C25_embedding/C25_embedding_weights/strided_slice/stack_1Const*
dtype0*
valueB:*
_output_shapes
:
Ѓ
dinput_layer/sparse_input_layer/input_layer/C25_embedding/C25_embedding_weights/strided_slice/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
Є
\input_layer/sparse_input_layer/input_layer/C25_embedding/C25_embedding_weights/strided_sliceStridedSliceTinput_layer/sparse_input_layer/input_layer/C25_embedding/C25_embedding_weights/Shapebinput_layer/sparse_input_layer/input_layer/C25_embedding/C25_embedding_weights/strided_slice/stackdinput_layer/sparse_input_layer/input_layer/C25_embedding/C25_embedding_weights/strided_slice/stack_1dinput_layer/sparse_input_layer/input_layer/C25_embedding/C25_embedding_weights/strided_slice/stack_2*
shrink_axis_mask*
Index0*
T0*
_output_shapes
: 
Ш
Vinput_layer/sparse_input_layer/input_layer/C25_embedding/C25_embedding_weights/stack/0Const*
_output_shapes
: *
value	B :*
dtype0
ј
Tinput_layer/sparse_input_layer/input_layer/C25_embedding/C25_embedding_weights/stackPackVinput_layer/sparse_input_layer/input_layer/C25_embedding/C25_embedding_weights/stack/0\input_layer/sparse_input_layer/input_layer/C25_embedding/C25_embedding_weights/strided_slice*
T0*
_output_shapes
:*
N
∆
Sinput_layer/sparse_input_layer/input_layer/C25_embedding/C25_embedding_weights/TileTileXinput_layer/sparse_input_layer/input_layer/C25_embedding/C25_embedding_weights/Reshape_1Tinput_layer/sparse_input_layer/input_layer/C25_embedding/C25_embedding_weights/stack*
T0
*0
_output_shapes
:€€€€€€€€€€€€€€€€€€
А
Yinput_layer/sparse_input_layer/input_layer/C25_embedding/C25_embedding_weights/zeros_like	ZerosLikefinput_layer/sparse_input_layer/input_layer/C25_embedding/C25_embedding_weights/embedding_lookup_sparse*
T0*'
_output_shapes
:€€€€€€€€€
Ґ
Ninput_layer/sparse_input_layer/input_layer/C25_embedding/C25_embedding_weightsSelectSinput_layer/sparse_input_layer/input_layer/C25_embedding/C25_embedding_weights/TileYinput_layer/sparse_input_layer/input_layer/C25_embedding/C25_embedding_weights/zeros_likefinput_layer/sparse_input_layer/input_layer/C25_embedding/C25_embedding_weights/embedding_lookup_sparse*'
_output_shapes
:€€€€€€€€€*
T0
з
Uinput_layer/sparse_input_layer/input_layer/C25_embedding/C25_embedding_weights/Cast_1CastTinput_layer/sparse_input_layer/input_layer/C25_embedding/to_sparse_input/dense_shape*
_output_shapes
:*

DstT0*

SrcT0	
¶
\input_layer/sparse_input_layer/input_layer/C25_embedding/C25_embedding_weights/Slice_1/beginConst*
dtype0*
valueB: *
_output_shapes
:
•
[input_layer/sparse_input_layer/input_layer/C25_embedding/C25_embedding_weights/Slice_1/sizeConst*
dtype0*
valueB:*
_output_shapes
:
£
Vinput_layer/sparse_input_layer/input_layer/C25_embedding/C25_embedding_weights/Slice_1SliceUinput_layer/sparse_input_layer/input_layer/C25_embedding/C25_embedding_weights/Cast_1\input_layer/sparse_input_layer/input_layer/C25_embedding/C25_embedding_weights/Slice_1/begin[input_layer/sparse_input_layer/input_layer/C25_embedding/C25_embedding_weights/Slice_1/size*
_output_shapes
:*
T0*
Index0
‘
Vinput_layer/sparse_input_layer/input_layer/C25_embedding/C25_embedding_weights/Shape_1ShapeNinput_layer/sparse_input_layer/input_layer/C25_embedding/C25_embedding_weights*
T0*
_output_shapes
:
¶
\input_layer/sparse_input_layer/input_layer/C25_embedding/C25_embedding_weights/Slice_2/beginConst*
dtype0*
_output_shapes
:*
valueB:
Ѓ
[input_layer/sparse_input_layer/input_layer/C25_embedding/C25_embedding_weights/Slice_2/sizeConst*
dtype0*
_output_shapes
:*
valueB:
€€€€€€€€€
§
Vinput_layer/sparse_input_layer/input_layer/C25_embedding/C25_embedding_weights/Slice_2SliceVinput_layer/sparse_input_layer/input_layer/C25_embedding/C25_embedding_weights/Shape_1\input_layer/sparse_input_layer/input_layer/C25_embedding/C25_embedding_weights/Slice_2/begin[input_layer/sparse_input_layer/input_layer/C25_embedding/C25_embedding_weights/Slice_2/size*
_output_shapes
:*
T0*
Index0
Ь
Zinput_layer/sparse_input_layer/input_layer/C25_embedding/C25_embedding_weights/concat/axisConst*
value	B : *
_output_shapes
: *
dtype0
Ы
Uinput_layer/sparse_input_layer/input_layer/C25_embedding/C25_embedding_weights/concatConcatV2Vinput_layer/sparse_input_layer/input_layer/C25_embedding/C25_embedding_weights/Slice_1Vinput_layer/sparse_input_layer/input_layer/C25_embedding/C25_embedding_weights/Slice_2Zinput_layer/sparse_input_layer/input_layer/C25_embedding/C25_embedding_weights/concat/axis*
_output_shapes
:*
N*
T0
Љ
Xinput_layer/sparse_input_layer/input_layer/C25_embedding/C25_embedding_weights/Reshape_2ReshapeNinput_layer/sparse_input_layer/input_layer/C25_embedding/C25_embedding_weightsUinput_layer/sparse_input_layer/input_layer/C25_embedding/C25_embedding_weights/concat*'
_output_shapes
:€€€€€€€€€*
T0
∆
>input_layer/sparse_input_layer/input_layer/C25_embedding/ShapeShapeXinput_layer/sparse_input_layer/input_layer/C25_embedding/C25_embedding_weights/Reshape_2*
_output_shapes
:*
T0
Ц
Linput_layer/sparse_input_layer/input_layer/C25_embedding/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Ш
Ninput_layer/sparse_input_layer/input_layer/C25_embedding/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Ш
Ninput_layer/sparse_input_layer/input_layer/C25_embedding/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
 
Finput_layer/sparse_input_layer/input_layer/C25_embedding/strided_sliceStridedSlice>input_layer/sparse_input_layer/input_layer/C25_embedding/ShapeLinput_layer/sparse_input_layer/input_layer/C25_embedding/strided_slice/stackNinput_layer/sparse_input_layer/input_layer/C25_embedding/strided_slice/stack_1Ninput_layer/sparse_input_layer/input_layer/C25_embedding/strided_slice/stack_2*
shrink_axis_mask*
Index0*
_output_shapes
: *
T0
К
Hinput_layer/sparse_input_layer/input_layer/C25_embedding/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
О
Finput_layer/sparse_input_layer/input_layer/C25_embedding/Reshape/shapePackFinput_layer/sparse_input_layer/input_layer/C25_embedding/strided_sliceHinput_layer/sparse_input_layer/input_layer/C25_embedding/Reshape/shape/1*
_output_shapes
:*
T0*
N
Я
@input_layer/sparse_input_layer/input_layer/C25_embedding/ReshapeReshapeXinput_layer/sparse_input_layer/input_layer/C25_embedding/C25_embedding_weights/Reshape_2Finput_layer/sparse_input_layer/input_layer/C25_embedding/Reshape/shape*'
_output_shapes
:€€€€€€€€€*
T0
Т
Ginput_layer/sparse_input_layer/input_layer/C26_embedding/ExpandDims/dimConst*
dtype0*
_output_shapes
: *
valueB :
€€€€€€€€€
№
Cinput_layer/sparse_input_layer/input_layer/C26_embedding/ExpandDims
ExpandDimsPlaceholder_38Ginput_layer/sparse_input_layer/input_layer/C26_embedding/ExpandDims/dim*
T0*'
_output_shapes
:€€€€€€€€€
Ш
Winput_layer/sparse_input_layer/input_layer/C26_embedding/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
valueB B *
dtype0
≠
Qinput_layer/sparse_input_layer/input_layer/C26_embedding/to_sparse_input/NotEqualNotEqualCinput_layer/sparse_input_layer/input_layer/C26_embedding/ExpandDimsWinput_layer/sparse_input_layer/input_layer/C26_embedding/to_sparse_input/ignore_value/x*'
_output_shapes
:€€€€€€€€€*
T0
’
Pinput_layer/sparse_input_layer/input_layer/C26_embedding/to_sparse_input/indicesWhereQinput_layer/sparse_input_layer/input_layer/C26_embedding/to_sparse_input/NotEqual*'
_output_shapes
:€€€€€€€€€
ґ
Oinput_layer/sparse_input_layer/input_layer/C26_embedding/to_sparse_input/valuesGatherNdCinput_layer/sparse_input_layer/input_layer/C26_embedding/ExpandDimsPinput_layer/sparse_input_layer/input_layer/C26_embedding/to_sparse_input/indices*#
_output_shapes
:€€€€€€€€€*
Tparams0*
Tindices0	
„
Tinput_layer/sparse_input_layer/input_layer/C26_embedding/to_sparse_input/dense_shapeShapeCinput_layer/sparse_input_layer/input_layer/C26_embedding/ExpandDims*
_output_shapes
:*
out_type0	*
T0
г
?input_layer/sparse_input_layer/input_layer/C26_embedding/lookupStringToHashBucketFastOinput_layer/sparse_input_layer/input_layer/C26_embedding/to_sparse_input/values*
num_bucketsРN*#
_output_shapes
:€€€€€€€€€
Э
minput_layer/sparse_input_layer/input_layer/C26_embedding/embedding_weights/Initializer/truncated_normal/shapeConst*
valueB"'     *
dtype0*
_output_shapes
:*]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C26_embedding/embedding_weights
Р
linput_layer/sparse_input_layer/input_layer/C26_embedding/embedding_weights/Initializer/truncated_normal/meanConst*]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C26_embedding/embedding_weights*
valueB
 *    *
dtype0*
_output_shapes
: 
Т
ninput_layer/sparse_input_layer/input_layer/C26_embedding/embedding_weights/Initializer/truncated_normal/stddevConst*
dtype0*]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C26_embedding/embedding_weights*
valueB
 *  А>*
_output_shapes
: 
П
winput_layer/sparse_input_layer/input_layer/C26_embedding/embedding_weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormalminput_layer/sparse_input_layer/input_layer/C26_embedding/embedding_weights/Initializer/truncated_normal/shape*
T0*]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C26_embedding/embedding_weights*
_output_shapes
:	РN*
dtype0
д
kinput_layer/sparse_input_layer/input_layer/C26_embedding/embedding_weights/Initializer/truncated_normal/mulMulwinput_layer/sparse_input_layer/input_layer/C26_embedding/embedding_weights/Initializer/truncated_normal/TruncatedNormalninput_layer/sparse_input_layer/input_layer/C26_embedding/embedding_weights/Initializer/truncated_normal/stddev*
_output_shapes
:	РN*
T0*]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C26_embedding/embedding_weights
“
ginput_layer/sparse_input_layer/input_layer/C26_embedding/embedding_weights/Initializer/truncated_normalAddkinput_layer/sparse_input_layer/input_layer/C26_embedding/embedding_weights/Initializer/truncated_normal/mullinput_layer/sparse_input_layer/input_layer/C26_embedding/embedding_weights/Initializer/truncated_normal/mean*
_output_shapes
:	РN*
T0*]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C26_embedding/embedding_weights
ы
Jinput_layer/sparse_input_layer/input_layer/C26_embedding/embedding_weights
VariableV2*]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C26_embedding/embedding_weights*
shape:	РN*
_output_shapes
:	РN*
dtype0
Щ
Qinput_layer/sparse_input_layer/input_layer/C26_embedding/embedding_weights/AssignAssignJinput_layer/sparse_input_layer/input_layer/C26_embedding/embedding_weightsginput_layer/sparse_input_layer/input_layer/C26_embedding/embedding_weights/Initializer/truncated_normal*
_output_shapes
:	РN*
T0*]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C26_embedding/embedding_weights
∞
Oinput_layer/sparse_input_layer/input_layer/C26_embedding/embedding_weights/readIdentityJinput_layer/sparse_input_layer/input_layer/C26_embedding/embedding_weights*
_output_shapes
:	РN*
T0*]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C26_embedding/embedding_weights
§
Zinput_layer/sparse_input_layer/input_layer/C26_embedding/C26_embedding_weights/Slice/beginConst*
_output_shapes
:*
valueB: *
dtype0
£
Yinput_layer/sparse_input_layer/input_layer/C26_embedding/C26_embedding_weights/Slice/sizeConst*
_output_shapes
:*
dtype0*
valueB:
Ь
Tinput_layer/sparse_input_layer/input_layer/C26_embedding/C26_embedding_weights/SliceSliceTinput_layer/sparse_input_layer/input_layer/C26_embedding/to_sparse_input/dense_shapeZinput_layer/sparse_input_layer/input_layer/C26_embedding/C26_embedding_weights/Slice/beginYinput_layer/sparse_input_layer/input_layer/C26_embedding/C26_embedding_weights/Slice/size*
T0	*
Index0*
_output_shapes
:
Ю
Tinput_layer/sparse_input_layer/input_layer/C26_embedding/C26_embedding_weights/ConstConst*
dtype0*
valueB: *
_output_shapes
:
®
Sinput_layer/sparse_input_layer/input_layer/C26_embedding/C26_embedding_weights/ProdProdTinput_layer/sparse_input_layer/input_layer/C26_embedding/C26_embedding_weights/SliceTinput_layer/sparse_input_layer/input_layer/C26_embedding/C26_embedding_weights/Const*
T0	*
_output_shapes
: 
°
_input_layer/sparse_input_layer/input_layer/C26_embedding/C26_embedding_weights/GatherV2/indicesConst*
dtype0*
value	B :*
_output_shapes
: 
Ю
\input_layer/sparse_input_layer/input_layer/C26_embedding/C26_embedding_weights/GatherV2/axisConst*
value	B : *
_output_shapes
: *
dtype0
Љ
Winput_layer/sparse_input_layer/input_layer/C26_embedding/C26_embedding_weights/GatherV2GatherV2Tinput_layer/sparse_input_layer/input_layer/C26_embedding/to_sparse_input/dense_shape_input_layer/sparse_input_layer/input_layer/C26_embedding/C26_embedding_weights/GatherV2/indices\input_layer/sparse_input_layer/input_layer/C26_embedding/C26_embedding_weights/GatherV2/axis*
Taxis0*
Tparams0	*
Tindices0*
_output_shapes
: 
є
Uinput_layer/sparse_input_layer/input_layer/C26_embedding/C26_embedding_weights/Cast/xPackSinput_layer/sparse_input_layer/input_layer/C26_embedding/C26_embedding_weights/ProdWinput_layer/sparse_input_layer/input_layer/C26_embedding/C26_embedding_weights/GatherV2*
_output_shapes
:*
N*
T0	
Ы
\input_layer/sparse_input_layer/input_layer/C26_embedding/C26_embedding_weights/SparseReshapeSparseReshapePinput_layer/sparse_input_layer/input_layer/C26_embedding/to_sparse_input/indicesTinput_layer/sparse_input_layer/input_layer/C26_embedding/to_sparse_input/dense_shapeUinput_layer/sparse_input_layer/input_layer/C26_embedding/C26_embedding_weights/Cast/x*-
_output_shapes
:€€€€€€€€€:
а
einput_layer/sparse_input_layer/input_layer/C26_embedding/C26_embedding_weights/SparseReshape/IdentityIdentity?input_layer/sparse_input_layer/input_layer/C26_embedding/lookup*
T0	*#
_output_shapes
:€€€€€€€€€
Я
]input_layer/sparse_input_layer/input_layer/C26_embedding/C26_embedding_weights/GreaterEqual/yConst*
value	B	 R *
dtype0	*
_output_shapes
: 
я
[input_layer/sparse_input_layer/input_layer/C26_embedding/C26_embedding_weights/GreaterEqualGreaterEqualeinput_layer/sparse_input_layer/input_layer/C26_embedding/C26_embedding_weights/SparseReshape/Identity]input_layer/sparse_input_layer/input_layer/C26_embedding/C26_embedding_weights/GreaterEqual/y*#
_output_shapes
:€€€€€€€€€*
T0	
г
Tinput_layer/sparse_input_layer/input_layer/C26_embedding/C26_embedding_weights/WhereWhere[input_layer/sparse_input_layer/input_layer/C26_embedding/C26_embedding_weights/GreaterEqual*'
_output_shapes
:€€€€€€€€€
ѓ
\input_layer/sparse_input_layer/input_layer/C26_embedding/C26_embedding_weights/Reshape/shapeConst*
_output_shapes
:*
valueB:
€€€€€€€€€*
dtype0
√
Vinput_layer/sparse_input_layer/input_layer/C26_embedding/C26_embedding_weights/ReshapeReshapeTinput_layer/sparse_input_layer/input_layer/C26_embedding/C26_embedding_weights/Where\input_layer/sparse_input_layer/input_layer/C26_embedding/C26_embedding_weights/Reshape/shape*#
_output_shapes
:€€€€€€€€€*
T0	
†
^input_layer/sparse_input_layer/input_layer/C26_embedding/C26_embedding_weights/GatherV2_1/axisConst*
dtype0*
value	B : *
_output_shapes
: 
–
Yinput_layer/sparse_input_layer/input_layer/C26_embedding/C26_embedding_weights/GatherV2_1GatherV2\input_layer/sparse_input_layer/input_layer/C26_embedding/C26_embedding_weights/SparseReshapeVinput_layer/sparse_input_layer/input_layer/C26_embedding/C26_embedding_weights/Reshape^input_layer/sparse_input_layer/input_layer/C26_embedding/C26_embedding_weights/GatherV2_1/axis*
Tparams0	*
Taxis0*
Tindices0	*'
_output_shapes
:€€€€€€€€€
†
^input_layer/sparse_input_layer/input_layer/C26_embedding/C26_embedding_weights/GatherV2_2/axisConst*
dtype0*
value	B : *
_output_shapes
: 
’
Yinput_layer/sparse_input_layer/input_layer/C26_embedding/C26_embedding_weights/GatherV2_2GatherV2einput_layer/sparse_input_layer/input_layer/C26_embedding/C26_embedding_weights/SparseReshape/IdentityVinput_layer/sparse_input_layer/input_layer/C26_embedding/C26_embedding_weights/Reshape^input_layer/sparse_input_layer/input_layer/C26_embedding/C26_embedding_weights/GatherV2_2/axis*#
_output_shapes
:€€€€€€€€€*
Taxis0*
Tparams0	*
Tindices0	
и
Winput_layer/sparse_input_layer/input_layer/C26_embedding/C26_embedding_weights/IdentityIdentity^input_layer/sparse_input_layer/input_layer/C26_embedding/C26_embedding_weights/SparseReshape:1*
T0	*
_output_shapes
:
™
hinput_layer/sparse_input_layer/input_layer/C26_embedding/C26_embedding_weights/SparseFillEmptyRows/ConstConst*
dtype0	*
value	B	 R *
_output_shapes
: 
е
vinput_layer/sparse_input_layer/input_layer/C26_embedding/C26_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowsSparseFillEmptyRowsYinput_layer/sparse_input_layer/input_layer/C26_embedding/C26_embedding_weights/GatherV2_1Yinput_layer/sparse_input_layer/input_layer/C26_embedding/C26_embedding_weights/GatherV2_2Winput_layer/sparse_input_layer/input_layer/C26_embedding/C26_embedding_weights/Identityhinput_layer/sparse_input_layer/input_layer/C26_embedding/C26_embedding_weights/SparseFillEmptyRows/Const*
T0	*T
_output_shapesB
@:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€
Ћ
zinput_layer/sparse_input_layer/input_layer/C26_embedding/C26_embedding_weights/embedding_lookup_sparse/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        
Ќ
|input_layer/sparse_input_layer/input_layer/C26_embedding/C26_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB"       
Ќ
|input_layer/sparse_input_layer/input_layer/C26_embedding/C26_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
й
tinput_layer/sparse_input_layer/input_layer/C26_embedding/C26_embedding_weights/embedding_lookup_sparse/strided_sliceStridedSlicevinput_layer/sparse_input_layer/input_layer/C26_embedding/C26_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowszinput_layer/sparse_input_layer/input_layer/C26_embedding/C26_embedding_weights/embedding_lookup_sparse/strided_slice/stack|input_layer/sparse_input_layer/input_layer/C26_embedding/C26_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1|input_layer/sparse_input_layer/input_layer/C26_embedding/C26_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2*
shrink_axis_mask*
Index0*
end_mask*#
_output_shapes
:€€€€€€€€€*
T0	*

begin_mask
¶
kinput_layer/sparse_input_layer/input_layer/C26_embedding/C26_embedding_weights/embedding_lookup_sparse/CastCasttinput_layer/sparse_input_layer/input_layer/C26_embedding/C26_embedding_weights/embedding_lookup_sparse/strided_slice*

SrcT0	*#
_output_shapes
:€€€€€€€€€*

DstT0
Ѓ
minput_layer/sparse_input_layer/input_layer/C26_embedding/C26_embedding_weights/embedding_lookup_sparse/UniqueUniquexinput_layer/sparse_input_layer/input_layer/C26_embedding/C26_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:1*
T0	*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
Э
|input_layer/sparse_input_layer/input_layer/C26_embedding/C26_embedding_weights/embedding_lookup_sparse/embedding_lookup/axisConst*
value	B : *]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C26_embedding/embedding_weights*
_output_shapes
: *
dtype0
х
winput_layer/sparse_input_layer/input_layer/C26_embedding/C26_embedding_weights/embedding_lookup_sparse/embedding_lookupGatherV2Oinput_layer/sparse_input_layer/input_layer/C26_embedding/embedding_weights/readminput_layer/sparse_input_layer/input_layer/C26_embedding/C26_embedding_weights/embedding_lookup_sparse/Unique|input_layer/sparse_input_layer/input_layer/C26_embedding/C26_embedding_weights/embedding_lookup_sparse/embedding_lookup/axis*
Tindices0	*
Tparams0*
Taxis0*]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C26_embedding/embedding_weights*'
_output_shapes
:€€€€€€€€€
Є
Аinput_layer/sparse_input_layer/input_layer/C26_embedding/C26_embedding_weights/embedding_lookup_sparse/embedding_lookup/IdentityIdentitywinput_layer/sparse_input_layer/input_layer/C26_embedding/C26_embedding_weights/embedding_lookup_sparse/embedding_lookup*
T0*'
_output_shapes
:€€€€€€€€€
О
finput_layer/sparse_input_layer/input_layer/C26_embedding/C26_embedding_weights/embedding_lookup_sparseSparseSegmentMeanАinput_layer/sparse_input_layer/input_layer/C26_embedding/C26_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identityoinput_layer/sparse_input_layer/input_layer/C26_embedding/C26_embedding_weights/embedding_lookup_sparse/Unique:1kinput_layer/sparse_input_layer/input_layer/C26_embedding/C26_embedding_weights/embedding_lookup_sparse/Cast*'
_output_shapes
:€€€€€€€€€*
T0
ѓ
^input_layer/sparse_input_layer/input_layer/C26_embedding/C26_embedding_weights/Reshape_1/shapeConst*
_output_shapes
:*
valueB"€€€€   *
dtype0
п
Xinput_layer/sparse_input_layer/input_layer/C26_embedding/C26_embedding_weights/Reshape_1Reshapexinput_layer/sparse_input_layer/input_layer/C26_embedding/C26_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:2^input_layer/sparse_input_layer/input_layer/C26_embedding/C26_embedding_weights/Reshape_1/shape*
T0
*'
_output_shapes
:€€€€€€€€€
к
Tinput_layer/sparse_input_layer/input_layer/C26_embedding/C26_embedding_weights/ShapeShapefinput_layer/sparse_input_layer/input_layer/C26_embedding/C26_embedding_weights/embedding_lookup_sparse*
_output_shapes
:*
T0
ђ
binput_layer/sparse_input_layer/input_layer/C26_embedding/C26_embedding_weights/strided_slice/stackConst*
valueB:*
_output_shapes
:*
dtype0
Ѓ
dinput_layer/sparse_input_layer/input_layer/C26_embedding/C26_embedding_weights/strided_slice/stack_1Const*
valueB:*
_output_shapes
:*
dtype0
Ѓ
dinput_layer/sparse_input_layer/input_layer/C26_embedding/C26_embedding_weights/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
Є
\input_layer/sparse_input_layer/input_layer/C26_embedding/C26_embedding_weights/strided_sliceStridedSliceTinput_layer/sparse_input_layer/input_layer/C26_embedding/C26_embedding_weights/Shapebinput_layer/sparse_input_layer/input_layer/C26_embedding/C26_embedding_weights/strided_slice/stackdinput_layer/sparse_input_layer/input_layer/C26_embedding/C26_embedding_weights/strided_slice/stack_1dinput_layer/sparse_input_layer/input_layer/C26_embedding/C26_embedding_weights/strided_slice/stack_2*
shrink_axis_mask*
Index0*
T0*
_output_shapes
: 
Ш
Vinput_layer/sparse_input_layer/input_layer/C26_embedding/C26_embedding_weights/stack/0Const*
value	B :*
dtype0*
_output_shapes
: 
ј
Tinput_layer/sparse_input_layer/input_layer/C26_embedding/C26_embedding_weights/stackPackVinput_layer/sparse_input_layer/input_layer/C26_embedding/C26_embedding_weights/stack/0\input_layer/sparse_input_layer/input_layer/C26_embedding/C26_embedding_weights/strided_slice*
T0*
N*
_output_shapes
:
∆
Sinput_layer/sparse_input_layer/input_layer/C26_embedding/C26_embedding_weights/TileTileXinput_layer/sparse_input_layer/input_layer/C26_embedding/C26_embedding_weights/Reshape_1Tinput_layer/sparse_input_layer/input_layer/C26_embedding/C26_embedding_weights/stack*
T0
*0
_output_shapes
:€€€€€€€€€€€€€€€€€€
А
Yinput_layer/sparse_input_layer/input_layer/C26_embedding/C26_embedding_weights/zeros_like	ZerosLikefinput_layer/sparse_input_layer/input_layer/C26_embedding/C26_embedding_weights/embedding_lookup_sparse*
T0*'
_output_shapes
:€€€€€€€€€
Ґ
Ninput_layer/sparse_input_layer/input_layer/C26_embedding/C26_embedding_weightsSelectSinput_layer/sparse_input_layer/input_layer/C26_embedding/C26_embedding_weights/TileYinput_layer/sparse_input_layer/input_layer/C26_embedding/C26_embedding_weights/zeros_likefinput_layer/sparse_input_layer/input_layer/C26_embedding/C26_embedding_weights/embedding_lookup_sparse*'
_output_shapes
:€€€€€€€€€*
T0
з
Uinput_layer/sparse_input_layer/input_layer/C26_embedding/C26_embedding_weights/Cast_1CastTinput_layer/sparse_input_layer/input_layer/C26_embedding/to_sparse_input/dense_shape*

SrcT0	*

DstT0*
_output_shapes
:
¶
\input_layer/sparse_input_layer/input_layer/C26_embedding/C26_embedding_weights/Slice_1/beginConst*
dtype0*
_output_shapes
:*
valueB: 
•
[input_layer/sparse_input_layer/input_layer/C26_embedding/C26_embedding_weights/Slice_1/sizeConst*
_output_shapes
:*
valueB:*
dtype0
£
Vinput_layer/sparse_input_layer/input_layer/C26_embedding/C26_embedding_weights/Slice_1SliceUinput_layer/sparse_input_layer/input_layer/C26_embedding/C26_embedding_weights/Cast_1\input_layer/sparse_input_layer/input_layer/C26_embedding/C26_embedding_weights/Slice_1/begin[input_layer/sparse_input_layer/input_layer/C26_embedding/C26_embedding_weights/Slice_1/size*
T0*
Index0*
_output_shapes
:
‘
Vinput_layer/sparse_input_layer/input_layer/C26_embedding/C26_embedding_weights/Shape_1ShapeNinput_layer/sparse_input_layer/input_layer/C26_embedding/C26_embedding_weights*
T0*
_output_shapes
:
¶
\input_layer/sparse_input_layer/input_layer/C26_embedding/C26_embedding_weights/Slice_2/beginConst*
_output_shapes
:*
dtype0*
valueB:
Ѓ
[input_layer/sparse_input_layer/input_layer/C26_embedding/C26_embedding_weights/Slice_2/sizeConst*
dtype0*
valueB:
€€€€€€€€€*
_output_shapes
:
§
Vinput_layer/sparse_input_layer/input_layer/C26_embedding/C26_embedding_weights/Slice_2SliceVinput_layer/sparse_input_layer/input_layer/C26_embedding/C26_embedding_weights/Shape_1\input_layer/sparse_input_layer/input_layer/C26_embedding/C26_embedding_weights/Slice_2/begin[input_layer/sparse_input_layer/input_layer/C26_embedding/C26_embedding_weights/Slice_2/size*
_output_shapes
:*
Index0*
T0
Ь
Zinput_layer/sparse_input_layer/input_layer/C26_embedding/C26_embedding_weights/concat/axisConst*
dtype0*
_output_shapes
: *
value	B : 
Ы
Uinput_layer/sparse_input_layer/input_layer/C26_embedding/C26_embedding_weights/concatConcatV2Vinput_layer/sparse_input_layer/input_layer/C26_embedding/C26_embedding_weights/Slice_1Vinput_layer/sparse_input_layer/input_layer/C26_embedding/C26_embedding_weights/Slice_2Zinput_layer/sparse_input_layer/input_layer/C26_embedding/C26_embedding_weights/concat/axis*
_output_shapes
:*
N*
T0
Љ
Xinput_layer/sparse_input_layer/input_layer/C26_embedding/C26_embedding_weights/Reshape_2ReshapeNinput_layer/sparse_input_layer/input_layer/C26_embedding/C26_embedding_weightsUinput_layer/sparse_input_layer/input_layer/C26_embedding/C26_embedding_weights/concat*'
_output_shapes
:€€€€€€€€€*
T0
∆
>input_layer/sparse_input_layer/input_layer/C26_embedding/ShapeShapeXinput_layer/sparse_input_layer/input_layer/C26_embedding/C26_embedding_weights/Reshape_2*
_output_shapes
:*
T0
Ц
Linput_layer/sparse_input_layer/input_layer/C26_embedding/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Ш
Ninput_layer/sparse_input_layer/input_layer/C26_embedding/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
Ш
Ninput_layer/sparse_input_layer/input_layer/C26_embedding/strided_slice/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
 
Finput_layer/sparse_input_layer/input_layer/C26_embedding/strided_sliceStridedSlice>input_layer/sparse_input_layer/input_layer/C26_embedding/ShapeLinput_layer/sparse_input_layer/input_layer/C26_embedding/strided_slice/stackNinput_layer/sparse_input_layer/input_layer/C26_embedding/strided_slice/stack_1Ninput_layer/sparse_input_layer/input_layer/C26_embedding/strided_slice/stack_2*
shrink_axis_mask*
T0*
Index0*
_output_shapes
: 
К
Hinput_layer/sparse_input_layer/input_layer/C26_embedding/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
О
Finput_layer/sparse_input_layer/input_layer/C26_embedding/Reshape/shapePackFinput_layer/sparse_input_layer/input_layer/C26_embedding/strided_sliceHinput_layer/sparse_input_layer/input_layer/C26_embedding/Reshape/shape/1*
N*
_output_shapes
:*
T0
Я
@input_layer/sparse_input_layer/input_layer/C26_embedding/ReshapeReshapeXinput_layer/sparse_input_layer/input_layer/C26_embedding/C26_embedding_weights/Reshape_2Finput_layer/sparse_input_layer/input_layer/C26_embedding/Reshape/shape*'
_output_shapes
:€€€€€€€€€*
T0
С
Finput_layer/sparse_input_layer/input_layer/C2_embedding/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€
Џ
Binput_layer/sparse_input_layer/input_layer/C2_embedding/ExpandDims
ExpandDimsPlaceholder_14Finput_layer/sparse_input_layer/input_layer/C2_embedding/ExpandDims/dim*
T0*'
_output_shapes
:€€€€€€€€€
Ч
Vinput_layer/sparse_input_layer/input_layer/C2_embedding/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB B 
™
Pinput_layer/sparse_input_layer/input_layer/C2_embedding/to_sparse_input/NotEqualNotEqualBinput_layer/sparse_input_layer/input_layer/C2_embedding/ExpandDimsVinput_layer/sparse_input_layer/input_layer/C2_embedding/to_sparse_input/ignore_value/x*
T0*'
_output_shapes
:€€€€€€€€€
”
Oinput_layer/sparse_input_layer/input_layer/C2_embedding/to_sparse_input/indicesWherePinput_layer/sparse_input_layer/input_layer/C2_embedding/to_sparse_input/NotEqual*'
_output_shapes
:€€€€€€€€€
≥
Ninput_layer/sparse_input_layer/input_layer/C2_embedding/to_sparse_input/valuesGatherNdBinput_layer/sparse_input_layer/input_layer/C2_embedding/ExpandDimsOinput_layer/sparse_input_layer/input_layer/C2_embedding/to_sparse_input/indices*
Tindices0	*
Tparams0*#
_output_shapes
:€€€€€€€€€
’
Sinput_layer/sparse_input_layer/input_layer/C2_embedding/to_sparse_input/dense_shapeShapeBinput_layer/sparse_input_layer/input_layer/C2_embedding/ExpandDims*
out_type0	*
T0*
_output_shapes
:
б
>input_layer/sparse_input_layer/input_layer/C2_embedding/lookupStringToHashBucketFastNinput_layer/sparse_input_layer/input_layer/C2_embedding/to_sparse_input/values*
num_bucketsРN*#
_output_shapes
:€€€€€€€€€
Ы
linput_layer/sparse_input_layer/input_layer/C2_embedding/embedding_weights/Initializer/truncated_normal/shapeConst*\
_classR
PNloc:@input_layer/sparse_input_layer/input_layer/C2_embedding/embedding_weights*
dtype0*
valueB"'     *
_output_shapes
:
О
kinput_layer/sparse_input_layer/input_layer/C2_embedding/embedding_weights/Initializer/truncated_normal/meanConst*
valueB
 *    *
_output_shapes
: *
dtype0*\
_classR
PNloc:@input_layer/sparse_input_layer/input_layer/C2_embedding/embedding_weights
Р
minput_layer/sparse_input_layer/input_layer/C2_embedding/embedding_weights/Initializer/truncated_normal/stddevConst*
dtype0*
_output_shapes
: *\
_classR
PNloc:@input_layer/sparse_input_layer/input_layer/C2_embedding/embedding_weights*
valueB
 *  А>
М
vinput_layer/sparse_input_layer/input_layer/C2_embedding/embedding_weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormallinput_layer/sparse_input_layer/input_layer/C2_embedding/embedding_weights/Initializer/truncated_normal/shape*
T0*\
_classR
PNloc:@input_layer/sparse_input_layer/input_layer/C2_embedding/embedding_weights*
dtype0*
_output_shapes
:	РN
а
jinput_layer/sparse_input_layer/input_layer/C2_embedding/embedding_weights/Initializer/truncated_normal/mulMulvinput_layer/sparse_input_layer/input_layer/C2_embedding/embedding_weights/Initializer/truncated_normal/TruncatedNormalminput_layer/sparse_input_layer/input_layer/C2_embedding/embedding_weights/Initializer/truncated_normal/stddev*
T0*
_output_shapes
:	РN*\
_classR
PNloc:@input_layer/sparse_input_layer/input_layer/C2_embedding/embedding_weights
ќ
finput_layer/sparse_input_layer/input_layer/C2_embedding/embedding_weights/Initializer/truncated_normalAddjinput_layer/sparse_input_layer/input_layer/C2_embedding/embedding_weights/Initializer/truncated_normal/mulkinput_layer/sparse_input_layer/input_layer/C2_embedding/embedding_weights/Initializer/truncated_normal/mean*
T0*
_output_shapes
:	РN*\
_classR
PNloc:@input_layer/sparse_input_layer/input_layer/C2_embedding/embedding_weights
щ
Iinput_layer/sparse_input_layer/input_layer/C2_embedding/embedding_weights
VariableV2*
shape:	РN*
_output_shapes
:	РN*\
_classR
PNloc:@input_layer/sparse_input_layer/input_layer/C2_embedding/embedding_weights*
dtype0
Х
Pinput_layer/sparse_input_layer/input_layer/C2_embedding/embedding_weights/AssignAssignIinput_layer/sparse_input_layer/input_layer/C2_embedding/embedding_weightsfinput_layer/sparse_input_layer/input_layer/C2_embedding/embedding_weights/Initializer/truncated_normal*
T0*\
_classR
PNloc:@input_layer/sparse_input_layer/input_layer/C2_embedding/embedding_weights*
_output_shapes
:	РN
≠
Ninput_layer/sparse_input_layer/input_layer/C2_embedding/embedding_weights/readIdentityIinput_layer/sparse_input_layer/input_layer/C2_embedding/embedding_weights*\
_classR
PNloc:@input_layer/sparse_input_layer/input_layer/C2_embedding/embedding_weights*
T0*
_output_shapes
:	РN
Ґ
Xinput_layer/sparse_input_layer/input_layer/C2_embedding/C2_embedding_weights/Slice/beginConst*
valueB: *
_output_shapes
:*
dtype0
°
Winput_layer/sparse_input_layer/input_layer/C2_embedding/C2_embedding_weights/Slice/sizeConst*
valueB:*
_output_shapes
:*
dtype0
Х
Rinput_layer/sparse_input_layer/input_layer/C2_embedding/C2_embedding_weights/SliceSliceSinput_layer/sparse_input_layer/input_layer/C2_embedding/to_sparse_input/dense_shapeXinput_layer/sparse_input_layer/input_layer/C2_embedding/C2_embedding_weights/Slice/beginWinput_layer/sparse_input_layer/input_layer/C2_embedding/C2_embedding_weights/Slice/size*
Index0*
T0	*
_output_shapes
:
Ь
Rinput_layer/sparse_input_layer/input_layer/C2_embedding/C2_embedding_weights/ConstConst*
_output_shapes
:*
valueB: *
dtype0
Ґ
Qinput_layer/sparse_input_layer/input_layer/C2_embedding/C2_embedding_weights/ProdProdRinput_layer/sparse_input_layer/input_layer/C2_embedding/C2_embedding_weights/SliceRinput_layer/sparse_input_layer/input_layer/C2_embedding/C2_embedding_weights/Const*
T0	*
_output_shapes
: 
Я
]input_layer/sparse_input_layer/input_layer/C2_embedding/C2_embedding_weights/GatherV2/indicesConst*
value	B :*
_output_shapes
: *
dtype0
Ь
Zinput_layer/sparse_input_layer/input_layer/C2_embedding/C2_embedding_weights/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 
µ
Uinput_layer/sparse_input_layer/input_layer/C2_embedding/C2_embedding_weights/GatherV2GatherV2Sinput_layer/sparse_input_layer/input_layer/C2_embedding/to_sparse_input/dense_shape]input_layer/sparse_input_layer/input_layer/C2_embedding/C2_embedding_weights/GatherV2/indicesZinput_layer/sparse_input_layer/input_layer/C2_embedding/C2_embedding_weights/GatherV2/axis*
Tindices0*
Taxis0*
_output_shapes
: *
Tparams0	
≥
Sinput_layer/sparse_input_layer/input_layer/C2_embedding/C2_embedding_weights/Cast/xPackQinput_layer/sparse_input_layer/input_layer/C2_embedding/C2_embedding_weights/ProdUinput_layer/sparse_input_layer/input_layer/C2_embedding/C2_embedding_weights/GatherV2*
T0	*
_output_shapes
:*
N
Х
Zinput_layer/sparse_input_layer/input_layer/C2_embedding/C2_embedding_weights/SparseReshapeSparseReshapeOinput_layer/sparse_input_layer/input_layer/C2_embedding/to_sparse_input/indicesSinput_layer/sparse_input_layer/input_layer/C2_embedding/to_sparse_input/dense_shapeSinput_layer/sparse_input_layer/input_layer/C2_embedding/C2_embedding_weights/Cast/x*-
_output_shapes
:€€€€€€€€€:
Ё
cinput_layer/sparse_input_layer/input_layer/C2_embedding/C2_embedding_weights/SparseReshape/IdentityIdentity>input_layer/sparse_input_layer/input_layer/C2_embedding/lookup*#
_output_shapes
:€€€€€€€€€*
T0	
Э
[input_layer/sparse_input_layer/input_layer/C2_embedding/C2_embedding_weights/GreaterEqual/yConst*
_output_shapes
: *
dtype0	*
value	B	 R 
ў
Yinput_layer/sparse_input_layer/input_layer/C2_embedding/C2_embedding_weights/GreaterEqualGreaterEqualcinput_layer/sparse_input_layer/input_layer/C2_embedding/C2_embedding_weights/SparseReshape/Identity[input_layer/sparse_input_layer/input_layer/C2_embedding/C2_embedding_weights/GreaterEqual/y*#
_output_shapes
:€€€€€€€€€*
T0	
я
Rinput_layer/sparse_input_layer/input_layer/C2_embedding/C2_embedding_weights/WhereWhereYinput_layer/sparse_input_layer/input_layer/C2_embedding/C2_embedding_weights/GreaterEqual*'
_output_shapes
:€€€€€€€€€
≠
Zinput_layer/sparse_input_layer/input_layer/C2_embedding/C2_embedding_weights/Reshape/shapeConst*
_output_shapes
:*
valueB:
€€€€€€€€€*
dtype0
љ
Tinput_layer/sparse_input_layer/input_layer/C2_embedding/C2_embedding_weights/ReshapeReshapeRinput_layer/sparse_input_layer/input_layer/C2_embedding/C2_embedding_weights/WhereZinput_layer/sparse_input_layer/input_layer/C2_embedding/C2_embedding_weights/Reshape/shape*#
_output_shapes
:€€€€€€€€€*
T0	
Ю
\input_layer/sparse_input_layer/input_layer/C2_embedding/C2_embedding_weights/GatherV2_1/axisConst*
dtype0*
value	B : *
_output_shapes
: 
»
Winput_layer/sparse_input_layer/input_layer/C2_embedding/C2_embedding_weights/GatherV2_1GatherV2Zinput_layer/sparse_input_layer/input_layer/C2_embedding/C2_embedding_weights/SparseReshapeTinput_layer/sparse_input_layer/input_layer/C2_embedding/C2_embedding_weights/Reshape\input_layer/sparse_input_layer/input_layer/C2_embedding/C2_embedding_weights/GatherV2_1/axis*
Tindices0	*
Tparams0	*
Taxis0*'
_output_shapes
:€€€€€€€€€
Ю
\input_layer/sparse_input_layer/input_layer/C2_embedding/C2_embedding_weights/GatherV2_2/axisConst*
value	B : *
dtype0*
_output_shapes
: 
Ќ
Winput_layer/sparse_input_layer/input_layer/C2_embedding/C2_embedding_weights/GatherV2_2GatherV2cinput_layer/sparse_input_layer/input_layer/C2_embedding/C2_embedding_weights/SparseReshape/IdentityTinput_layer/sparse_input_layer/input_layer/C2_embedding/C2_embedding_weights/Reshape\input_layer/sparse_input_layer/input_layer/C2_embedding/C2_embedding_weights/GatherV2_2/axis*
Tparams0	*
Taxis0*#
_output_shapes
:€€€€€€€€€*
Tindices0	
д
Uinput_layer/sparse_input_layer/input_layer/C2_embedding/C2_embedding_weights/IdentityIdentity\input_layer/sparse_input_layer/input_layer/C2_embedding/C2_embedding_weights/SparseReshape:1*
_output_shapes
:*
T0	
®
finput_layer/sparse_input_layer/input_layer/C2_embedding/C2_embedding_weights/SparseFillEmptyRows/ConstConst*
dtype0	*
value	B	 R *
_output_shapes
: 
џ
tinput_layer/sparse_input_layer/input_layer/C2_embedding/C2_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowsSparseFillEmptyRowsWinput_layer/sparse_input_layer/input_layer/C2_embedding/C2_embedding_weights/GatherV2_1Winput_layer/sparse_input_layer/input_layer/C2_embedding/C2_embedding_weights/GatherV2_2Uinput_layer/sparse_input_layer/input_layer/C2_embedding/C2_embedding_weights/Identityfinput_layer/sparse_input_layer/input_layer/C2_embedding/C2_embedding_weights/SparseFillEmptyRows/Const*
T0	*T
_output_shapesB
@:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€
…
xinput_layer/sparse_input_layer/input_layer/C2_embedding/C2_embedding_weights/embedding_lookup_sparse/strided_slice/stackConst*
_output_shapes
:*
valueB"        *
dtype0
Ћ
zinput_layer/sparse_input_layer/input_layer/C2_embedding/C2_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1Const*
dtype0*
valueB"       *
_output_shapes
:
Ћ
zinput_layer/sparse_input_layer/input_layer/C2_embedding/C2_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:
я
rinput_layer/sparse_input_layer/input_layer/C2_embedding/C2_embedding_weights/embedding_lookup_sparse/strided_sliceStridedSlicetinput_layer/sparse_input_layer/input_layer/C2_embedding/C2_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowsxinput_layer/sparse_input_layer/input_layer/C2_embedding/C2_embedding_weights/embedding_lookup_sparse/strided_slice/stackzinput_layer/sparse_input_layer/input_layer/C2_embedding/C2_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1zinput_layer/sparse_input_layer/input_layer/C2_embedding/C2_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2*
T0	*
shrink_axis_mask*
Index0*
end_mask*#
_output_shapes
:€€€€€€€€€*

begin_mask
Ґ
iinput_layer/sparse_input_layer/input_layer/C2_embedding/C2_embedding_weights/embedding_lookup_sparse/CastCastrinput_layer/sparse_input_layer/input_layer/C2_embedding/C2_embedding_weights/embedding_lookup_sparse/strided_slice*

SrcT0	*

DstT0*#
_output_shapes
:€€€€€€€€€
™
kinput_layer/sparse_input_layer/input_layer/C2_embedding/C2_embedding_weights/embedding_lookup_sparse/UniqueUniquevinput_layer/sparse_input_layer/input_layer/C2_embedding/C2_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€*
T0	
Ъ
zinput_layer/sparse_input_layer/input_layer/C2_embedding/C2_embedding_weights/embedding_lookup_sparse/embedding_lookup/axisConst*
_output_shapes
: *
value	B : *\
_classR
PNloc:@input_layer/sparse_input_layer/input_layer/C2_embedding/embedding_weights*
dtype0
н
uinput_layer/sparse_input_layer/input_layer/C2_embedding/C2_embedding_weights/embedding_lookup_sparse/embedding_lookupGatherV2Ninput_layer/sparse_input_layer/input_layer/C2_embedding/embedding_weights/readkinput_layer/sparse_input_layer/input_layer/C2_embedding/C2_embedding_weights/embedding_lookup_sparse/Uniquezinput_layer/sparse_input_layer/input_layer/C2_embedding/C2_embedding_weights/embedding_lookup_sparse/embedding_lookup/axis*'
_output_shapes
:€€€€€€€€€*
Tindices0	*\
_classR
PNloc:@input_layer/sparse_input_layer/input_layer/C2_embedding/embedding_weights*
Tparams0*
Taxis0
≥
~input_layer/sparse_input_layer/input_layer/C2_embedding/C2_embedding_weights/embedding_lookup_sparse/embedding_lookup/IdentityIdentityuinput_layer/sparse_input_layer/input_layer/C2_embedding/C2_embedding_weights/embedding_lookup_sparse/embedding_lookup*'
_output_shapes
:€€€€€€€€€*
T0
Е
dinput_layer/sparse_input_layer/input_layer/C2_embedding/C2_embedding_weights/embedding_lookup_sparseSparseSegmentMean~input_layer/sparse_input_layer/input_layer/C2_embedding/C2_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identityminput_layer/sparse_input_layer/input_layer/C2_embedding/C2_embedding_weights/embedding_lookup_sparse/Unique:1iinput_layer/sparse_input_layer/input_layer/C2_embedding/C2_embedding_weights/embedding_lookup_sparse/Cast*'
_output_shapes
:€€€€€€€€€*
T0
≠
\input_layer/sparse_input_layer/input_layer/C2_embedding/C2_embedding_weights/Reshape_1/shapeConst*
_output_shapes
:*
valueB"€€€€   *
dtype0
й
Vinput_layer/sparse_input_layer/input_layer/C2_embedding/C2_embedding_weights/Reshape_1Reshapevinput_layer/sparse_input_layer/input_layer/C2_embedding/C2_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:2\input_layer/sparse_input_layer/input_layer/C2_embedding/C2_embedding_weights/Reshape_1/shape*'
_output_shapes
:€€€€€€€€€*
T0

ж
Rinput_layer/sparse_input_layer/input_layer/C2_embedding/C2_embedding_weights/ShapeShapedinput_layer/sparse_input_layer/input_layer/C2_embedding/C2_embedding_weights/embedding_lookup_sparse*
T0*
_output_shapes
:
™
`input_layer/sparse_input_layer/input_layer/C2_embedding/C2_embedding_weights/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
ђ
binput_layer/sparse_input_layer/input_layer/C2_embedding/C2_embedding_weights/strided_slice/stack_1Const*
dtype0*
valueB:*
_output_shapes
:
ђ
binput_layer/sparse_input_layer/input_layer/C2_embedding/C2_embedding_weights/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
Ѓ
Zinput_layer/sparse_input_layer/input_layer/C2_embedding/C2_embedding_weights/strided_sliceStridedSliceRinput_layer/sparse_input_layer/input_layer/C2_embedding/C2_embedding_weights/Shape`input_layer/sparse_input_layer/input_layer/C2_embedding/C2_embedding_weights/strided_slice/stackbinput_layer/sparse_input_layer/input_layer/C2_embedding/C2_embedding_weights/strided_slice/stack_1binput_layer/sparse_input_layer/input_layer/C2_embedding/C2_embedding_weights/strided_slice/stack_2*
_output_shapes
: *
shrink_axis_mask*
Index0*
T0
Ц
Tinput_layer/sparse_input_layer/input_layer/C2_embedding/C2_embedding_weights/stack/0Const*
value	B :*
_output_shapes
: *
dtype0
Ї
Rinput_layer/sparse_input_layer/input_layer/C2_embedding/C2_embedding_weights/stackPackTinput_layer/sparse_input_layer/input_layer/C2_embedding/C2_embedding_weights/stack/0Zinput_layer/sparse_input_layer/input_layer/C2_embedding/C2_embedding_weights/strided_slice*
T0*
_output_shapes
:*
N
ј
Qinput_layer/sparse_input_layer/input_layer/C2_embedding/C2_embedding_weights/TileTileVinput_layer/sparse_input_layer/input_layer/C2_embedding/C2_embedding_weights/Reshape_1Rinput_layer/sparse_input_layer/input_layer/C2_embedding/C2_embedding_weights/stack*0
_output_shapes
:€€€€€€€€€€€€€€€€€€*
T0

ь
Winput_layer/sparse_input_layer/input_layer/C2_embedding/C2_embedding_weights/zeros_like	ZerosLikedinput_layer/sparse_input_layer/input_layer/C2_embedding/C2_embedding_weights/embedding_lookup_sparse*'
_output_shapes
:€€€€€€€€€*
T0
Ъ
Linput_layer/sparse_input_layer/input_layer/C2_embedding/C2_embedding_weightsSelectQinput_layer/sparse_input_layer/input_layer/C2_embedding/C2_embedding_weights/TileWinput_layer/sparse_input_layer/input_layer/C2_embedding/C2_embedding_weights/zeros_likedinput_layer/sparse_input_layer/input_layer/C2_embedding/C2_embedding_weights/embedding_lookup_sparse*
T0*'
_output_shapes
:€€€€€€€€€
д
Sinput_layer/sparse_input_layer/input_layer/C2_embedding/C2_embedding_weights/Cast_1CastSinput_layer/sparse_input_layer/input_layer/C2_embedding/to_sparse_input/dense_shape*
_output_shapes
:*

SrcT0	*

DstT0
§
Zinput_layer/sparse_input_layer/input_layer/C2_embedding/C2_embedding_weights/Slice_1/beginConst*
valueB: *
dtype0*
_output_shapes
:
£
Yinput_layer/sparse_input_layer/input_layer/C2_embedding/C2_embedding_weights/Slice_1/sizeConst*
dtype0*
_output_shapes
:*
valueB:
Ы
Tinput_layer/sparse_input_layer/input_layer/C2_embedding/C2_embedding_weights/Slice_1SliceSinput_layer/sparse_input_layer/input_layer/C2_embedding/C2_embedding_weights/Cast_1Zinput_layer/sparse_input_layer/input_layer/C2_embedding/C2_embedding_weights/Slice_1/beginYinput_layer/sparse_input_layer/input_layer/C2_embedding/C2_embedding_weights/Slice_1/size*
_output_shapes
:*
T0*
Index0
–
Tinput_layer/sparse_input_layer/input_layer/C2_embedding/C2_embedding_weights/Shape_1ShapeLinput_layer/sparse_input_layer/input_layer/C2_embedding/C2_embedding_weights*
T0*
_output_shapes
:
§
Zinput_layer/sparse_input_layer/input_layer/C2_embedding/C2_embedding_weights/Slice_2/beginConst*
_output_shapes
:*
dtype0*
valueB:
ђ
Yinput_layer/sparse_input_layer/input_layer/C2_embedding/C2_embedding_weights/Slice_2/sizeConst*
valueB:
€€€€€€€€€*
dtype0*
_output_shapes
:
Ь
Tinput_layer/sparse_input_layer/input_layer/C2_embedding/C2_embedding_weights/Slice_2SliceTinput_layer/sparse_input_layer/input_layer/C2_embedding/C2_embedding_weights/Shape_1Zinput_layer/sparse_input_layer/input_layer/C2_embedding/C2_embedding_weights/Slice_2/beginYinput_layer/sparse_input_layer/input_layer/C2_embedding/C2_embedding_weights/Slice_2/size*
T0*
_output_shapes
:*
Index0
Ъ
Xinput_layer/sparse_input_layer/input_layer/C2_embedding/C2_embedding_weights/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
У
Sinput_layer/sparse_input_layer/input_layer/C2_embedding/C2_embedding_weights/concatConcatV2Tinput_layer/sparse_input_layer/input_layer/C2_embedding/C2_embedding_weights/Slice_1Tinput_layer/sparse_input_layer/input_layer/C2_embedding/C2_embedding_weights/Slice_2Xinput_layer/sparse_input_layer/input_layer/C2_embedding/C2_embedding_weights/concat/axis*
N*
T0*
_output_shapes
:
ґ
Vinput_layer/sparse_input_layer/input_layer/C2_embedding/C2_embedding_weights/Reshape_2ReshapeLinput_layer/sparse_input_layer/input_layer/C2_embedding/C2_embedding_weightsSinput_layer/sparse_input_layer/input_layer/C2_embedding/C2_embedding_weights/concat*'
_output_shapes
:€€€€€€€€€*
T0
√
=input_layer/sparse_input_layer/input_layer/C2_embedding/ShapeShapeVinput_layer/sparse_input_layer/input_layer/C2_embedding/C2_embedding_weights/Reshape_2*
T0*
_output_shapes
:
Х
Kinput_layer/sparse_input_layer/input_layer/C2_embedding/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Ч
Minput_layer/sparse_input_layer/input_layer/C2_embedding/strided_slice/stack_1Const*
_output_shapes
:*
valueB:*
dtype0
Ч
Minput_layer/sparse_input_layer/input_layer/C2_embedding/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
≈
Einput_layer/sparse_input_layer/input_layer/C2_embedding/strided_sliceStridedSlice=input_layer/sparse_input_layer/input_layer/C2_embedding/ShapeKinput_layer/sparse_input_layer/input_layer/C2_embedding/strided_slice/stackMinput_layer/sparse_input_layer/input_layer/C2_embedding/strided_slice/stack_1Minput_layer/sparse_input_layer/input_layer/C2_embedding/strided_slice/stack_2*
shrink_axis_mask*
_output_shapes
: *
Index0*
T0
Й
Ginput_layer/sparse_input_layer/input_layer/C2_embedding/Reshape/shape/1Const*
value	B :*
dtype0*
_output_shapes
: 
Л
Einput_layer/sparse_input_layer/input_layer/C2_embedding/Reshape/shapePackEinput_layer/sparse_input_layer/input_layer/C2_embedding/strided_sliceGinput_layer/sparse_input_layer/input_layer/C2_embedding/Reshape/shape/1*
N*
_output_shapes
:*
T0
Ы
?input_layer/sparse_input_layer/input_layer/C2_embedding/ReshapeReshapeVinput_layer/sparse_input_layer/input_layer/C2_embedding/C2_embedding_weights/Reshape_2Einput_layer/sparse_input_layer/input_layer/C2_embedding/Reshape/shape*'
_output_shapes
:€€€€€€€€€*
T0
С
Finput_layer/sparse_input_layer/input_layer/C3_embedding/ExpandDims/dimConst*
valueB :
€€€€€€€€€*
_output_shapes
: *
dtype0
Џ
Binput_layer/sparse_input_layer/input_layer/C3_embedding/ExpandDims
ExpandDimsPlaceholder_15Finput_layer/sparse_input_layer/input_layer/C3_embedding/ExpandDims/dim*
T0*'
_output_shapes
:€€€€€€€€€
Ч
Vinput_layer/sparse_input_layer/input_layer/C3_embedding/to_sparse_input/ignore_value/xConst*
dtype0*
_output_shapes
: *
valueB B 
™
Pinput_layer/sparse_input_layer/input_layer/C3_embedding/to_sparse_input/NotEqualNotEqualBinput_layer/sparse_input_layer/input_layer/C3_embedding/ExpandDimsVinput_layer/sparse_input_layer/input_layer/C3_embedding/to_sparse_input/ignore_value/x*'
_output_shapes
:€€€€€€€€€*
T0
”
Oinput_layer/sparse_input_layer/input_layer/C3_embedding/to_sparse_input/indicesWherePinput_layer/sparse_input_layer/input_layer/C3_embedding/to_sparse_input/NotEqual*'
_output_shapes
:€€€€€€€€€
≥
Ninput_layer/sparse_input_layer/input_layer/C3_embedding/to_sparse_input/valuesGatherNdBinput_layer/sparse_input_layer/input_layer/C3_embedding/ExpandDimsOinput_layer/sparse_input_layer/input_layer/C3_embedding/to_sparse_input/indices*#
_output_shapes
:€€€€€€€€€*
Tparams0*
Tindices0	
’
Sinput_layer/sparse_input_layer/input_layer/C3_embedding/to_sparse_input/dense_shapeShapeBinput_layer/sparse_input_layer/input_layer/C3_embedding/ExpandDims*
T0*
_output_shapes
:*
out_type0	
б
>input_layer/sparse_input_layer/input_layer/C3_embedding/lookupStringToHashBucketFastNinput_layer/sparse_input_layer/input_layer/C3_embedding/to_sparse_input/values*#
_output_shapes
:€€€€€€€€€*
num_bucketsРN
Ы
linput_layer/sparse_input_layer/input_layer/C3_embedding/embedding_weights/Initializer/truncated_normal/shapeConst*\
_classR
PNloc:@input_layer/sparse_input_layer/input_layer/C3_embedding/embedding_weights*
dtype0*
_output_shapes
:*
valueB"'     
О
kinput_layer/sparse_input_layer/input_layer/C3_embedding/embedding_weights/Initializer/truncated_normal/meanConst*
valueB
 *    *
_output_shapes
: *\
_classR
PNloc:@input_layer/sparse_input_layer/input_layer/C3_embedding/embedding_weights*
dtype0
Р
minput_layer/sparse_input_layer/input_layer/C3_embedding/embedding_weights/Initializer/truncated_normal/stddevConst*
_output_shapes
: *
valueB
 *  А>*
dtype0*\
_classR
PNloc:@input_layer/sparse_input_layer/input_layer/C3_embedding/embedding_weights
М
vinput_layer/sparse_input_layer/input_layer/C3_embedding/embedding_weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormallinput_layer/sparse_input_layer/input_layer/C3_embedding/embedding_weights/Initializer/truncated_normal/shape*\
_classR
PNloc:@input_layer/sparse_input_layer/input_layer/C3_embedding/embedding_weights*
T0*
_output_shapes
:	РN*
dtype0
а
jinput_layer/sparse_input_layer/input_layer/C3_embedding/embedding_weights/Initializer/truncated_normal/mulMulvinput_layer/sparse_input_layer/input_layer/C3_embedding/embedding_weights/Initializer/truncated_normal/TruncatedNormalminput_layer/sparse_input_layer/input_layer/C3_embedding/embedding_weights/Initializer/truncated_normal/stddev*
_output_shapes
:	РN*
T0*\
_classR
PNloc:@input_layer/sparse_input_layer/input_layer/C3_embedding/embedding_weights
ќ
finput_layer/sparse_input_layer/input_layer/C3_embedding/embedding_weights/Initializer/truncated_normalAddjinput_layer/sparse_input_layer/input_layer/C3_embedding/embedding_weights/Initializer/truncated_normal/mulkinput_layer/sparse_input_layer/input_layer/C3_embedding/embedding_weights/Initializer/truncated_normal/mean*\
_classR
PNloc:@input_layer/sparse_input_layer/input_layer/C3_embedding/embedding_weights*
T0*
_output_shapes
:	РN
щ
Iinput_layer/sparse_input_layer/input_layer/C3_embedding/embedding_weights
VariableV2*\
_classR
PNloc:@input_layer/sparse_input_layer/input_layer/C3_embedding/embedding_weights*
_output_shapes
:	РN*
dtype0*
shape:	РN
Х
Pinput_layer/sparse_input_layer/input_layer/C3_embedding/embedding_weights/AssignAssignIinput_layer/sparse_input_layer/input_layer/C3_embedding/embedding_weightsfinput_layer/sparse_input_layer/input_layer/C3_embedding/embedding_weights/Initializer/truncated_normal*
T0*\
_classR
PNloc:@input_layer/sparse_input_layer/input_layer/C3_embedding/embedding_weights*
_output_shapes
:	РN
≠
Ninput_layer/sparse_input_layer/input_layer/C3_embedding/embedding_weights/readIdentityIinput_layer/sparse_input_layer/input_layer/C3_embedding/embedding_weights*
_output_shapes
:	РN*
T0*\
_classR
PNloc:@input_layer/sparse_input_layer/input_layer/C3_embedding/embedding_weights
Ґ
Xinput_layer/sparse_input_layer/input_layer/C3_embedding/C3_embedding_weights/Slice/beginConst*
valueB: *
dtype0*
_output_shapes
:
°
Winput_layer/sparse_input_layer/input_layer/C3_embedding/C3_embedding_weights/Slice/sizeConst*
dtype0*
valueB:*
_output_shapes
:
Х
Rinput_layer/sparse_input_layer/input_layer/C3_embedding/C3_embedding_weights/SliceSliceSinput_layer/sparse_input_layer/input_layer/C3_embedding/to_sparse_input/dense_shapeXinput_layer/sparse_input_layer/input_layer/C3_embedding/C3_embedding_weights/Slice/beginWinput_layer/sparse_input_layer/input_layer/C3_embedding/C3_embedding_weights/Slice/size*
Index0*
T0	*
_output_shapes
:
Ь
Rinput_layer/sparse_input_layer/input_layer/C3_embedding/C3_embedding_weights/ConstConst*
valueB: *
_output_shapes
:*
dtype0
Ґ
Qinput_layer/sparse_input_layer/input_layer/C3_embedding/C3_embedding_weights/ProdProdRinput_layer/sparse_input_layer/input_layer/C3_embedding/C3_embedding_weights/SliceRinput_layer/sparse_input_layer/input_layer/C3_embedding/C3_embedding_weights/Const*
_output_shapes
: *
T0	
Я
]input_layer/sparse_input_layer/input_layer/C3_embedding/C3_embedding_weights/GatherV2/indicesConst*
_output_shapes
: *
value	B :*
dtype0
Ь
Zinput_layer/sparse_input_layer/input_layer/C3_embedding/C3_embedding_weights/GatherV2/axisConst*
dtype0*
_output_shapes
: *
value	B : 
µ
Uinput_layer/sparse_input_layer/input_layer/C3_embedding/C3_embedding_weights/GatherV2GatherV2Sinput_layer/sparse_input_layer/input_layer/C3_embedding/to_sparse_input/dense_shape]input_layer/sparse_input_layer/input_layer/C3_embedding/C3_embedding_weights/GatherV2/indicesZinput_layer/sparse_input_layer/input_layer/C3_embedding/C3_embedding_weights/GatherV2/axis*
Tindices0*
Taxis0*
Tparams0	*
_output_shapes
: 
≥
Sinput_layer/sparse_input_layer/input_layer/C3_embedding/C3_embedding_weights/Cast/xPackQinput_layer/sparse_input_layer/input_layer/C3_embedding/C3_embedding_weights/ProdUinput_layer/sparse_input_layer/input_layer/C3_embedding/C3_embedding_weights/GatherV2*
N*
T0	*
_output_shapes
:
Х
Zinput_layer/sparse_input_layer/input_layer/C3_embedding/C3_embedding_weights/SparseReshapeSparseReshapeOinput_layer/sparse_input_layer/input_layer/C3_embedding/to_sparse_input/indicesSinput_layer/sparse_input_layer/input_layer/C3_embedding/to_sparse_input/dense_shapeSinput_layer/sparse_input_layer/input_layer/C3_embedding/C3_embedding_weights/Cast/x*-
_output_shapes
:€€€€€€€€€:
Ё
cinput_layer/sparse_input_layer/input_layer/C3_embedding/C3_embedding_weights/SparseReshape/IdentityIdentity>input_layer/sparse_input_layer/input_layer/C3_embedding/lookup*
T0	*#
_output_shapes
:€€€€€€€€€
Э
[input_layer/sparse_input_layer/input_layer/C3_embedding/C3_embedding_weights/GreaterEqual/yConst*
value	B	 R *
dtype0	*
_output_shapes
: 
ў
Yinput_layer/sparse_input_layer/input_layer/C3_embedding/C3_embedding_weights/GreaterEqualGreaterEqualcinput_layer/sparse_input_layer/input_layer/C3_embedding/C3_embedding_weights/SparseReshape/Identity[input_layer/sparse_input_layer/input_layer/C3_embedding/C3_embedding_weights/GreaterEqual/y*
T0	*#
_output_shapes
:€€€€€€€€€
я
Rinput_layer/sparse_input_layer/input_layer/C3_embedding/C3_embedding_weights/WhereWhereYinput_layer/sparse_input_layer/input_layer/C3_embedding/C3_embedding_weights/GreaterEqual*'
_output_shapes
:€€€€€€€€€
≠
Zinput_layer/sparse_input_layer/input_layer/C3_embedding/C3_embedding_weights/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€
љ
Tinput_layer/sparse_input_layer/input_layer/C3_embedding/C3_embedding_weights/ReshapeReshapeRinput_layer/sparse_input_layer/input_layer/C3_embedding/C3_embedding_weights/WhereZinput_layer/sparse_input_layer/input_layer/C3_embedding/C3_embedding_weights/Reshape/shape*
T0	*#
_output_shapes
:€€€€€€€€€
Ю
\input_layer/sparse_input_layer/input_layer/C3_embedding/C3_embedding_weights/GatherV2_1/axisConst*
_output_shapes
: *
value	B : *
dtype0
»
Winput_layer/sparse_input_layer/input_layer/C3_embedding/C3_embedding_weights/GatherV2_1GatherV2Zinput_layer/sparse_input_layer/input_layer/C3_embedding/C3_embedding_weights/SparseReshapeTinput_layer/sparse_input_layer/input_layer/C3_embedding/C3_embedding_weights/Reshape\input_layer/sparse_input_layer/input_layer/C3_embedding/C3_embedding_weights/GatherV2_1/axis*
Tparams0	*
Tindices0	*
Taxis0*'
_output_shapes
:€€€€€€€€€
Ю
\input_layer/sparse_input_layer/input_layer/C3_embedding/C3_embedding_weights/GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
value	B : 
Ќ
Winput_layer/sparse_input_layer/input_layer/C3_embedding/C3_embedding_weights/GatherV2_2GatherV2cinput_layer/sparse_input_layer/input_layer/C3_embedding/C3_embedding_weights/SparseReshape/IdentityTinput_layer/sparse_input_layer/input_layer/C3_embedding/C3_embedding_weights/Reshape\input_layer/sparse_input_layer/input_layer/C3_embedding/C3_embedding_weights/GatherV2_2/axis*
Tparams0	*
Taxis0*
Tindices0	*#
_output_shapes
:€€€€€€€€€
д
Uinput_layer/sparse_input_layer/input_layer/C3_embedding/C3_embedding_weights/IdentityIdentity\input_layer/sparse_input_layer/input_layer/C3_embedding/C3_embedding_weights/SparseReshape:1*
_output_shapes
:*
T0	
®
finput_layer/sparse_input_layer/input_layer/C3_embedding/C3_embedding_weights/SparseFillEmptyRows/ConstConst*
_output_shapes
: *
value	B	 R *
dtype0	
џ
tinput_layer/sparse_input_layer/input_layer/C3_embedding/C3_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowsSparseFillEmptyRowsWinput_layer/sparse_input_layer/input_layer/C3_embedding/C3_embedding_weights/GatherV2_1Winput_layer/sparse_input_layer/input_layer/C3_embedding/C3_embedding_weights/GatherV2_2Uinput_layer/sparse_input_layer/input_layer/C3_embedding/C3_embedding_weights/Identityfinput_layer/sparse_input_layer/input_layer/C3_embedding/C3_embedding_weights/SparseFillEmptyRows/Const*T
_output_shapesB
@:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€*
T0	
…
xinput_layer/sparse_input_layer/input_layer/C3_embedding/C3_embedding_weights/embedding_lookup_sparse/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        
Ћ
zinput_layer/sparse_input_layer/input_layer/C3_embedding/C3_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1Const*
valueB"       *
_output_shapes
:*
dtype0
Ћ
zinput_layer/sparse_input_layer/input_layer/C3_embedding/C3_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:
я
rinput_layer/sparse_input_layer/input_layer/C3_embedding/C3_embedding_weights/embedding_lookup_sparse/strided_sliceStridedSlicetinput_layer/sparse_input_layer/input_layer/C3_embedding/C3_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowsxinput_layer/sparse_input_layer/input_layer/C3_embedding/C3_embedding_weights/embedding_lookup_sparse/strided_slice/stackzinput_layer/sparse_input_layer/input_layer/C3_embedding/C3_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1zinput_layer/sparse_input_layer/input_layer/C3_embedding/C3_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2*
end_mask*
T0	*

begin_mask*#
_output_shapes
:€€€€€€€€€*
Index0*
shrink_axis_mask
Ґ
iinput_layer/sparse_input_layer/input_layer/C3_embedding/C3_embedding_weights/embedding_lookup_sparse/CastCastrinput_layer/sparse_input_layer/input_layer/C3_embedding/C3_embedding_weights/embedding_lookup_sparse/strided_slice*#
_output_shapes
:€€€€€€€€€*

SrcT0	*

DstT0
™
kinput_layer/sparse_input_layer/input_layer/C3_embedding/C3_embedding_weights/embedding_lookup_sparse/UniqueUniquevinput_layer/sparse_input_layer/input_layer/C3_embedding/C3_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€*
T0	
Ъ
zinput_layer/sparse_input_layer/input_layer/C3_embedding/C3_embedding_weights/embedding_lookup_sparse/embedding_lookup/axisConst*
_output_shapes
: *
value	B : *
dtype0*\
_classR
PNloc:@input_layer/sparse_input_layer/input_layer/C3_embedding/embedding_weights
н
uinput_layer/sparse_input_layer/input_layer/C3_embedding/C3_embedding_weights/embedding_lookup_sparse/embedding_lookupGatherV2Ninput_layer/sparse_input_layer/input_layer/C3_embedding/embedding_weights/readkinput_layer/sparse_input_layer/input_layer/C3_embedding/C3_embedding_weights/embedding_lookup_sparse/Uniquezinput_layer/sparse_input_layer/input_layer/C3_embedding/C3_embedding_weights/embedding_lookup_sparse/embedding_lookup/axis*
Tparams0*
Taxis0*\
_classR
PNloc:@input_layer/sparse_input_layer/input_layer/C3_embedding/embedding_weights*'
_output_shapes
:€€€€€€€€€*
Tindices0	
≥
~input_layer/sparse_input_layer/input_layer/C3_embedding/C3_embedding_weights/embedding_lookup_sparse/embedding_lookup/IdentityIdentityuinput_layer/sparse_input_layer/input_layer/C3_embedding/C3_embedding_weights/embedding_lookup_sparse/embedding_lookup*'
_output_shapes
:€€€€€€€€€*
T0
Е
dinput_layer/sparse_input_layer/input_layer/C3_embedding/C3_embedding_weights/embedding_lookup_sparseSparseSegmentMean~input_layer/sparse_input_layer/input_layer/C3_embedding/C3_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identityminput_layer/sparse_input_layer/input_layer/C3_embedding/C3_embedding_weights/embedding_lookup_sparse/Unique:1iinput_layer/sparse_input_layer/input_layer/C3_embedding/C3_embedding_weights/embedding_lookup_sparse/Cast*'
_output_shapes
:€€€€€€€€€*
T0
≠
\input_layer/sparse_input_layer/input_layer/C3_embedding/C3_embedding_weights/Reshape_1/shapeConst*
dtype0*
_output_shapes
:*
valueB"€€€€   
й
Vinput_layer/sparse_input_layer/input_layer/C3_embedding/C3_embedding_weights/Reshape_1Reshapevinput_layer/sparse_input_layer/input_layer/C3_embedding/C3_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:2\input_layer/sparse_input_layer/input_layer/C3_embedding/C3_embedding_weights/Reshape_1/shape*
T0
*'
_output_shapes
:€€€€€€€€€
ж
Rinput_layer/sparse_input_layer/input_layer/C3_embedding/C3_embedding_weights/ShapeShapedinput_layer/sparse_input_layer/input_layer/C3_embedding/C3_embedding_weights/embedding_lookup_sparse*
_output_shapes
:*
T0
™
`input_layer/sparse_input_layer/input_layer/C3_embedding/C3_embedding_weights/strided_slice/stackConst*
_output_shapes
:*
valueB:*
dtype0
ђ
binput_layer/sparse_input_layer/input_layer/C3_embedding/C3_embedding_weights/strided_slice/stack_1Const*
valueB:*
_output_shapes
:*
dtype0
ђ
binput_layer/sparse_input_layer/input_layer/C3_embedding/C3_embedding_weights/strided_slice/stack_2Const*
valueB:*
_output_shapes
:*
dtype0
Ѓ
Zinput_layer/sparse_input_layer/input_layer/C3_embedding/C3_embedding_weights/strided_sliceStridedSliceRinput_layer/sparse_input_layer/input_layer/C3_embedding/C3_embedding_weights/Shape`input_layer/sparse_input_layer/input_layer/C3_embedding/C3_embedding_weights/strided_slice/stackbinput_layer/sparse_input_layer/input_layer/C3_embedding/C3_embedding_weights/strided_slice/stack_1binput_layer/sparse_input_layer/input_layer/C3_embedding/C3_embedding_weights/strided_slice/stack_2*
_output_shapes
: *
T0*
shrink_axis_mask*
Index0
Ц
Tinput_layer/sparse_input_layer/input_layer/C3_embedding/C3_embedding_weights/stack/0Const*
dtype0*
_output_shapes
: *
value	B :
Ї
Rinput_layer/sparse_input_layer/input_layer/C3_embedding/C3_embedding_weights/stackPackTinput_layer/sparse_input_layer/input_layer/C3_embedding/C3_embedding_weights/stack/0Zinput_layer/sparse_input_layer/input_layer/C3_embedding/C3_embedding_weights/strided_slice*
N*
T0*
_output_shapes
:
ј
Qinput_layer/sparse_input_layer/input_layer/C3_embedding/C3_embedding_weights/TileTileVinput_layer/sparse_input_layer/input_layer/C3_embedding/C3_embedding_weights/Reshape_1Rinput_layer/sparse_input_layer/input_layer/C3_embedding/C3_embedding_weights/stack*0
_output_shapes
:€€€€€€€€€€€€€€€€€€*
T0

ь
Winput_layer/sparse_input_layer/input_layer/C3_embedding/C3_embedding_weights/zeros_like	ZerosLikedinput_layer/sparse_input_layer/input_layer/C3_embedding/C3_embedding_weights/embedding_lookup_sparse*'
_output_shapes
:€€€€€€€€€*
T0
Ъ
Linput_layer/sparse_input_layer/input_layer/C3_embedding/C3_embedding_weightsSelectQinput_layer/sparse_input_layer/input_layer/C3_embedding/C3_embedding_weights/TileWinput_layer/sparse_input_layer/input_layer/C3_embedding/C3_embedding_weights/zeros_likedinput_layer/sparse_input_layer/input_layer/C3_embedding/C3_embedding_weights/embedding_lookup_sparse*'
_output_shapes
:€€€€€€€€€*
T0
д
Sinput_layer/sparse_input_layer/input_layer/C3_embedding/C3_embedding_weights/Cast_1CastSinput_layer/sparse_input_layer/input_layer/C3_embedding/to_sparse_input/dense_shape*

DstT0*
_output_shapes
:*

SrcT0	
§
Zinput_layer/sparse_input_layer/input_layer/C3_embedding/C3_embedding_weights/Slice_1/beginConst*
valueB: *
dtype0*
_output_shapes
:
£
Yinput_layer/sparse_input_layer/input_layer/C3_embedding/C3_embedding_weights/Slice_1/sizeConst*
valueB:*
_output_shapes
:*
dtype0
Ы
Tinput_layer/sparse_input_layer/input_layer/C3_embedding/C3_embedding_weights/Slice_1SliceSinput_layer/sparse_input_layer/input_layer/C3_embedding/C3_embedding_weights/Cast_1Zinput_layer/sparse_input_layer/input_layer/C3_embedding/C3_embedding_weights/Slice_1/beginYinput_layer/sparse_input_layer/input_layer/C3_embedding/C3_embedding_weights/Slice_1/size*
_output_shapes
:*
T0*
Index0
–
Tinput_layer/sparse_input_layer/input_layer/C3_embedding/C3_embedding_weights/Shape_1ShapeLinput_layer/sparse_input_layer/input_layer/C3_embedding/C3_embedding_weights*
_output_shapes
:*
T0
§
Zinput_layer/sparse_input_layer/input_layer/C3_embedding/C3_embedding_weights/Slice_2/beginConst*
dtype0*
valueB:*
_output_shapes
:
ђ
Yinput_layer/sparse_input_layer/input_layer/C3_embedding/C3_embedding_weights/Slice_2/sizeConst*
dtype0*
valueB:
€€€€€€€€€*
_output_shapes
:
Ь
Tinput_layer/sparse_input_layer/input_layer/C3_embedding/C3_embedding_weights/Slice_2SliceTinput_layer/sparse_input_layer/input_layer/C3_embedding/C3_embedding_weights/Shape_1Zinput_layer/sparse_input_layer/input_layer/C3_embedding/C3_embedding_weights/Slice_2/beginYinput_layer/sparse_input_layer/input_layer/C3_embedding/C3_embedding_weights/Slice_2/size*
_output_shapes
:*
T0*
Index0
Ъ
Xinput_layer/sparse_input_layer/input_layer/C3_embedding/C3_embedding_weights/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
У
Sinput_layer/sparse_input_layer/input_layer/C3_embedding/C3_embedding_weights/concatConcatV2Tinput_layer/sparse_input_layer/input_layer/C3_embedding/C3_embedding_weights/Slice_1Tinput_layer/sparse_input_layer/input_layer/C3_embedding/C3_embedding_weights/Slice_2Xinput_layer/sparse_input_layer/input_layer/C3_embedding/C3_embedding_weights/concat/axis*
T0*
N*
_output_shapes
:
ґ
Vinput_layer/sparse_input_layer/input_layer/C3_embedding/C3_embedding_weights/Reshape_2ReshapeLinput_layer/sparse_input_layer/input_layer/C3_embedding/C3_embedding_weightsSinput_layer/sparse_input_layer/input_layer/C3_embedding/C3_embedding_weights/concat*'
_output_shapes
:€€€€€€€€€*
T0
√
=input_layer/sparse_input_layer/input_layer/C3_embedding/ShapeShapeVinput_layer/sparse_input_layer/input_layer/C3_embedding/C3_embedding_weights/Reshape_2*
_output_shapes
:*
T0
Х
Kinput_layer/sparse_input_layer/input_layer/C3_embedding/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Ч
Minput_layer/sparse_input_layer/input_layer/C3_embedding/strided_slice/stack_1Const*
dtype0*
valueB:*
_output_shapes
:
Ч
Minput_layer/sparse_input_layer/input_layer/C3_embedding/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
≈
Einput_layer/sparse_input_layer/input_layer/C3_embedding/strided_sliceStridedSlice=input_layer/sparse_input_layer/input_layer/C3_embedding/ShapeKinput_layer/sparse_input_layer/input_layer/C3_embedding/strided_slice/stackMinput_layer/sparse_input_layer/input_layer/C3_embedding/strided_slice/stack_1Minput_layer/sparse_input_layer/input_layer/C3_embedding/strided_slice/stack_2*
T0*
shrink_axis_mask*
Index0*
_output_shapes
: 
Й
Ginput_layer/sparse_input_layer/input_layer/C3_embedding/Reshape/shape/1Const*
dtype0*
_output_shapes
: *
value	B :
Л
Einput_layer/sparse_input_layer/input_layer/C3_embedding/Reshape/shapePackEinput_layer/sparse_input_layer/input_layer/C3_embedding/strided_sliceGinput_layer/sparse_input_layer/input_layer/C3_embedding/Reshape/shape/1*
N*
T0*
_output_shapes
:
Ы
?input_layer/sparse_input_layer/input_layer/C3_embedding/ReshapeReshapeVinput_layer/sparse_input_layer/input_layer/C3_embedding/C3_embedding_weights/Reshape_2Einput_layer/sparse_input_layer/input_layer/C3_embedding/Reshape/shape*'
_output_shapes
:€€€€€€€€€*
T0
С
Finput_layer/sparse_input_layer/input_layer/C4_embedding/ExpandDims/dimConst*
valueB :
€€€€€€€€€*
_output_shapes
: *
dtype0
Џ
Binput_layer/sparse_input_layer/input_layer/C4_embedding/ExpandDims
ExpandDimsPlaceholder_16Finput_layer/sparse_input_layer/input_layer/C4_embedding/ExpandDims/dim*
T0*'
_output_shapes
:€€€€€€€€€
Ч
Vinput_layer/sparse_input_layer/input_layer/C4_embedding/to_sparse_input/ignore_value/xConst*
dtype0*
_output_shapes
: *
valueB B 
™
Pinput_layer/sparse_input_layer/input_layer/C4_embedding/to_sparse_input/NotEqualNotEqualBinput_layer/sparse_input_layer/input_layer/C4_embedding/ExpandDimsVinput_layer/sparse_input_layer/input_layer/C4_embedding/to_sparse_input/ignore_value/x*
T0*'
_output_shapes
:€€€€€€€€€
”
Oinput_layer/sparse_input_layer/input_layer/C4_embedding/to_sparse_input/indicesWherePinput_layer/sparse_input_layer/input_layer/C4_embedding/to_sparse_input/NotEqual*'
_output_shapes
:€€€€€€€€€
≥
Ninput_layer/sparse_input_layer/input_layer/C4_embedding/to_sparse_input/valuesGatherNdBinput_layer/sparse_input_layer/input_layer/C4_embedding/ExpandDimsOinput_layer/sparse_input_layer/input_layer/C4_embedding/to_sparse_input/indices*
Tparams0*
Tindices0	*#
_output_shapes
:€€€€€€€€€
’
Sinput_layer/sparse_input_layer/input_layer/C4_embedding/to_sparse_input/dense_shapeShapeBinput_layer/sparse_input_layer/input_layer/C4_embedding/ExpandDims*
out_type0	*
T0*
_output_shapes
:
б
>input_layer/sparse_input_layer/input_layer/C4_embedding/lookupStringToHashBucketFastNinput_layer/sparse_input_layer/input_layer/C4_embedding/to_sparse_input/values*#
_output_shapes
:€€€€€€€€€*
num_bucketsРN
Ы
linput_layer/sparse_input_layer/input_layer/C4_embedding/embedding_weights/Initializer/truncated_normal/shapeConst*\
_classR
PNloc:@input_layer/sparse_input_layer/input_layer/C4_embedding/embedding_weights*
valueB"'     *
dtype0*
_output_shapes
:
О
kinput_layer/sparse_input_layer/input_layer/C4_embedding/embedding_weights/Initializer/truncated_normal/meanConst*
dtype0*\
_classR
PNloc:@input_layer/sparse_input_layer/input_layer/C4_embedding/embedding_weights*
_output_shapes
: *
valueB
 *    
Р
minput_layer/sparse_input_layer/input_layer/C4_embedding/embedding_weights/Initializer/truncated_normal/stddevConst*\
_classR
PNloc:@input_layer/sparse_input_layer/input_layer/C4_embedding/embedding_weights*
valueB
 *  А>*
dtype0*
_output_shapes
: 
М
vinput_layer/sparse_input_layer/input_layer/C4_embedding/embedding_weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormallinput_layer/sparse_input_layer/input_layer/C4_embedding/embedding_weights/Initializer/truncated_normal/shape*\
_classR
PNloc:@input_layer/sparse_input_layer/input_layer/C4_embedding/embedding_weights*
dtype0*
_output_shapes
:	РN*
T0
а
jinput_layer/sparse_input_layer/input_layer/C4_embedding/embedding_weights/Initializer/truncated_normal/mulMulvinput_layer/sparse_input_layer/input_layer/C4_embedding/embedding_weights/Initializer/truncated_normal/TruncatedNormalminput_layer/sparse_input_layer/input_layer/C4_embedding/embedding_weights/Initializer/truncated_normal/stddev*
_output_shapes
:	РN*
T0*\
_classR
PNloc:@input_layer/sparse_input_layer/input_layer/C4_embedding/embedding_weights
ќ
finput_layer/sparse_input_layer/input_layer/C4_embedding/embedding_weights/Initializer/truncated_normalAddjinput_layer/sparse_input_layer/input_layer/C4_embedding/embedding_weights/Initializer/truncated_normal/mulkinput_layer/sparse_input_layer/input_layer/C4_embedding/embedding_weights/Initializer/truncated_normal/mean*
_output_shapes
:	РN*\
_classR
PNloc:@input_layer/sparse_input_layer/input_layer/C4_embedding/embedding_weights*
T0
щ
Iinput_layer/sparse_input_layer/input_layer/C4_embedding/embedding_weights
VariableV2*
_output_shapes
:	РN*
shape:	РN*\
_classR
PNloc:@input_layer/sparse_input_layer/input_layer/C4_embedding/embedding_weights*
dtype0
Х
Pinput_layer/sparse_input_layer/input_layer/C4_embedding/embedding_weights/AssignAssignIinput_layer/sparse_input_layer/input_layer/C4_embedding/embedding_weightsfinput_layer/sparse_input_layer/input_layer/C4_embedding/embedding_weights/Initializer/truncated_normal*
T0*
_output_shapes
:	РN*\
_classR
PNloc:@input_layer/sparse_input_layer/input_layer/C4_embedding/embedding_weights
≠
Ninput_layer/sparse_input_layer/input_layer/C4_embedding/embedding_weights/readIdentityIinput_layer/sparse_input_layer/input_layer/C4_embedding/embedding_weights*
T0*\
_classR
PNloc:@input_layer/sparse_input_layer/input_layer/C4_embedding/embedding_weights*
_output_shapes
:	РN
Ґ
Xinput_layer/sparse_input_layer/input_layer/C4_embedding/C4_embedding_weights/Slice/beginConst*
_output_shapes
:*
dtype0*
valueB: 
°
Winput_layer/sparse_input_layer/input_layer/C4_embedding/C4_embedding_weights/Slice/sizeConst*
dtype0*
_output_shapes
:*
valueB:
Х
Rinput_layer/sparse_input_layer/input_layer/C4_embedding/C4_embedding_weights/SliceSliceSinput_layer/sparse_input_layer/input_layer/C4_embedding/to_sparse_input/dense_shapeXinput_layer/sparse_input_layer/input_layer/C4_embedding/C4_embedding_weights/Slice/beginWinput_layer/sparse_input_layer/input_layer/C4_embedding/C4_embedding_weights/Slice/size*
T0	*
Index0*
_output_shapes
:
Ь
Rinput_layer/sparse_input_layer/input_layer/C4_embedding/C4_embedding_weights/ConstConst*
dtype0*
valueB: *
_output_shapes
:
Ґ
Qinput_layer/sparse_input_layer/input_layer/C4_embedding/C4_embedding_weights/ProdProdRinput_layer/sparse_input_layer/input_layer/C4_embedding/C4_embedding_weights/SliceRinput_layer/sparse_input_layer/input_layer/C4_embedding/C4_embedding_weights/Const*
_output_shapes
: *
T0	
Я
]input_layer/sparse_input_layer/input_layer/C4_embedding/C4_embedding_weights/GatherV2/indicesConst*
_output_shapes
: *
dtype0*
value	B :
Ь
Zinput_layer/sparse_input_layer/input_layer/C4_embedding/C4_embedding_weights/GatherV2/axisConst*
dtype0*
value	B : *
_output_shapes
: 
µ
Uinput_layer/sparse_input_layer/input_layer/C4_embedding/C4_embedding_weights/GatherV2GatherV2Sinput_layer/sparse_input_layer/input_layer/C4_embedding/to_sparse_input/dense_shape]input_layer/sparse_input_layer/input_layer/C4_embedding/C4_embedding_weights/GatherV2/indicesZinput_layer/sparse_input_layer/input_layer/C4_embedding/C4_embedding_weights/GatherV2/axis*
Taxis0*
Tparams0	*
Tindices0*
_output_shapes
: 
≥
Sinput_layer/sparse_input_layer/input_layer/C4_embedding/C4_embedding_weights/Cast/xPackQinput_layer/sparse_input_layer/input_layer/C4_embedding/C4_embedding_weights/ProdUinput_layer/sparse_input_layer/input_layer/C4_embedding/C4_embedding_weights/GatherV2*
T0	*
_output_shapes
:*
N
Х
Zinput_layer/sparse_input_layer/input_layer/C4_embedding/C4_embedding_weights/SparseReshapeSparseReshapeOinput_layer/sparse_input_layer/input_layer/C4_embedding/to_sparse_input/indicesSinput_layer/sparse_input_layer/input_layer/C4_embedding/to_sparse_input/dense_shapeSinput_layer/sparse_input_layer/input_layer/C4_embedding/C4_embedding_weights/Cast/x*-
_output_shapes
:€€€€€€€€€:
Ё
cinput_layer/sparse_input_layer/input_layer/C4_embedding/C4_embedding_weights/SparseReshape/IdentityIdentity>input_layer/sparse_input_layer/input_layer/C4_embedding/lookup*
T0	*#
_output_shapes
:€€€€€€€€€
Э
[input_layer/sparse_input_layer/input_layer/C4_embedding/C4_embedding_weights/GreaterEqual/yConst*
dtype0	*
_output_shapes
: *
value	B	 R 
ў
Yinput_layer/sparse_input_layer/input_layer/C4_embedding/C4_embedding_weights/GreaterEqualGreaterEqualcinput_layer/sparse_input_layer/input_layer/C4_embedding/C4_embedding_weights/SparseReshape/Identity[input_layer/sparse_input_layer/input_layer/C4_embedding/C4_embedding_weights/GreaterEqual/y*
T0	*#
_output_shapes
:€€€€€€€€€
я
Rinput_layer/sparse_input_layer/input_layer/C4_embedding/C4_embedding_weights/WhereWhereYinput_layer/sparse_input_layer/input_layer/C4_embedding/C4_embedding_weights/GreaterEqual*'
_output_shapes
:€€€€€€€€€
≠
Zinput_layer/sparse_input_layer/input_layer/C4_embedding/C4_embedding_weights/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€
љ
Tinput_layer/sparse_input_layer/input_layer/C4_embedding/C4_embedding_weights/ReshapeReshapeRinput_layer/sparse_input_layer/input_layer/C4_embedding/C4_embedding_weights/WhereZinput_layer/sparse_input_layer/input_layer/C4_embedding/C4_embedding_weights/Reshape/shape*
T0	*#
_output_shapes
:€€€€€€€€€
Ю
\input_layer/sparse_input_layer/input_layer/C4_embedding/C4_embedding_weights/GatherV2_1/axisConst*
dtype0*
_output_shapes
: *
value	B : 
»
Winput_layer/sparse_input_layer/input_layer/C4_embedding/C4_embedding_weights/GatherV2_1GatherV2Zinput_layer/sparse_input_layer/input_layer/C4_embedding/C4_embedding_weights/SparseReshapeTinput_layer/sparse_input_layer/input_layer/C4_embedding/C4_embedding_weights/Reshape\input_layer/sparse_input_layer/input_layer/C4_embedding/C4_embedding_weights/GatherV2_1/axis*'
_output_shapes
:€€€€€€€€€*
Taxis0*
Tparams0	*
Tindices0	
Ю
\input_layer/sparse_input_layer/input_layer/C4_embedding/C4_embedding_weights/GatherV2_2/axisConst*
_output_shapes
: *
value	B : *
dtype0
Ќ
Winput_layer/sparse_input_layer/input_layer/C4_embedding/C4_embedding_weights/GatherV2_2GatherV2cinput_layer/sparse_input_layer/input_layer/C4_embedding/C4_embedding_weights/SparseReshape/IdentityTinput_layer/sparse_input_layer/input_layer/C4_embedding/C4_embedding_weights/Reshape\input_layer/sparse_input_layer/input_layer/C4_embedding/C4_embedding_weights/GatherV2_2/axis*
Taxis0*
Tindices0	*
Tparams0	*#
_output_shapes
:€€€€€€€€€
д
Uinput_layer/sparse_input_layer/input_layer/C4_embedding/C4_embedding_weights/IdentityIdentity\input_layer/sparse_input_layer/input_layer/C4_embedding/C4_embedding_weights/SparseReshape:1*
_output_shapes
:*
T0	
®
finput_layer/sparse_input_layer/input_layer/C4_embedding/C4_embedding_weights/SparseFillEmptyRows/ConstConst*
dtype0	*
value	B	 R *
_output_shapes
: 
џ
tinput_layer/sparse_input_layer/input_layer/C4_embedding/C4_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowsSparseFillEmptyRowsWinput_layer/sparse_input_layer/input_layer/C4_embedding/C4_embedding_weights/GatherV2_1Winput_layer/sparse_input_layer/input_layer/C4_embedding/C4_embedding_weights/GatherV2_2Uinput_layer/sparse_input_layer/input_layer/C4_embedding/C4_embedding_weights/Identityfinput_layer/sparse_input_layer/input_layer/C4_embedding/C4_embedding_weights/SparseFillEmptyRows/Const*T
_output_shapesB
@:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€*
T0	
…
xinput_layer/sparse_input_layer/input_layer/C4_embedding/C4_embedding_weights/embedding_lookup_sparse/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB"        
Ћ
zinput_layer/sparse_input_layer/input_layer/C4_embedding/C4_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB"       
Ћ
zinput_layer/sparse_input_layer/input_layer/C4_embedding/C4_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
я
rinput_layer/sparse_input_layer/input_layer/C4_embedding/C4_embedding_weights/embedding_lookup_sparse/strided_sliceStridedSlicetinput_layer/sparse_input_layer/input_layer/C4_embedding/C4_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowsxinput_layer/sparse_input_layer/input_layer/C4_embedding/C4_embedding_weights/embedding_lookup_sparse/strided_slice/stackzinput_layer/sparse_input_layer/input_layer/C4_embedding/C4_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1zinput_layer/sparse_input_layer/input_layer/C4_embedding/C4_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2*
end_mask*

begin_mask*
shrink_axis_mask*#
_output_shapes
:€€€€€€€€€*
T0	*
Index0
Ґ
iinput_layer/sparse_input_layer/input_layer/C4_embedding/C4_embedding_weights/embedding_lookup_sparse/CastCastrinput_layer/sparse_input_layer/input_layer/C4_embedding/C4_embedding_weights/embedding_lookup_sparse/strided_slice*#
_output_shapes
:€€€€€€€€€*

SrcT0	*

DstT0
™
kinput_layer/sparse_input_layer/input_layer/C4_embedding/C4_embedding_weights/embedding_lookup_sparse/UniqueUniquevinput_layer/sparse_input_layer/input_layer/C4_embedding/C4_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:1*
T0	*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
Ъ
zinput_layer/sparse_input_layer/input_layer/C4_embedding/C4_embedding_weights/embedding_lookup_sparse/embedding_lookup/axisConst*
_output_shapes
: *\
_classR
PNloc:@input_layer/sparse_input_layer/input_layer/C4_embedding/embedding_weights*
dtype0*
value	B : 
н
uinput_layer/sparse_input_layer/input_layer/C4_embedding/C4_embedding_weights/embedding_lookup_sparse/embedding_lookupGatherV2Ninput_layer/sparse_input_layer/input_layer/C4_embedding/embedding_weights/readkinput_layer/sparse_input_layer/input_layer/C4_embedding/C4_embedding_weights/embedding_lookup_sparse/Uniquezinput_layer/sparse_input_layer/input_layer/C4_embedding/C4_embedding_weights/embedding_lookup_sparse/embedding_lookup/axis*
Taxis0*
Tparams0*
Tindices0	*'
_output_shapes
:€€€€€€€€€*\
_classR
PNloc:@input_layer/sparse_input_layer/input_layer/C4_embedding/embedding_weights
≥
~input_layer/sparse_input_layer/input_layer/C4_embedding/C4_embedding_weights/embedding_lookup_sparse/embedding_lookup/IdentityIdentityuinput_layer/sparse_input_layer/input_layer/C4_embedding/C4_embedding_weights/embedding_lookup_sparse/embedding_lookup*
T0*'
_output_shapes
:€€€€€€€€€
Е
dinput_layer/sparse_input_layer/input_layer/C4_embedding/C4_embedding_weights/embedding_lookup_sparseSparseSegmentMean~input_layer/sparse_input_layer/input_layer/C4_embedding/C4_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identityminput_layer/sparse_input_layer/input_layer/C4_embedding/C4_embedding_weights/embedding_lookup_sparse/Unique:1iinput_layer/sparse_input_layer/input_layer/C4_embedding/C4_embedding_weights/embedding_lookup_sparse/Cast*
T0*'
_output_shapes
:€€€€€€€€€
≠
\input_layer/sparse_input_layer/input_layer/C4_embedding/C4_embedding_weights/Reshape_1/shapeConst*
valueB"€€€€   *
dtype0*
_output_shapes
:
й
Vinput_layer/sparse_input_layer/input_layer/C4_embedding/C4_embedding_weights/Reshape_1Reshapevinput_layer/sparse_input_layer/input_layer/C4_embedding/C4_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:2\input_layer/sparse_input_layer/input_layer/C4_embedding/C4_embedding_weights/Reshape_1/shape*
T0
*'
_output_shapes
:€€€€€€€€€
ж
Rinput_layer/sparse_input_layer/input_layer/C4_embedding/C4_embedding_weights/ShapeShapedinput_layer/sparse_input_layer/input_layer/C4_embedding/C4_embedding_weights/embedding_lookup_sparse*
T0*
_output_shapes
:
™
`input_layer/sparse_input_layer/input_layer/C4_embedding/C4_embedding_weights/strided_slice/stackConst*
_output_shapes
:*
valueB:*
dtype0
ђ
binput_layer/sparse_input_layer/input_layer/C4_embedding/C4_embedding_weights/strided_slice/stack_1Const*
_output_shapes
:*
valueB:*
dtype0
ђ
binput_layer/sparse_input_layer/input_layer/C4_embedding/C4_embedding_weights/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Ѓ
Zinput_layer/sparse_input_layer/input_layer/C4_embedding/C4_embedding_weights/strided_sliceStridedSliceRinput_layer/sparse_input_layer/input_layer/C4_embedding/C4_embedding_weights/Shape`input_layer/sparse_input_layer/input_layer/C4_embedding/C4_embedding_weights/strided_slice/stackbinput_layer/sparse_input_layer/input_layer/C4_embedding/C4_embedding_weights/strided_slice/stack_1binput_layer/sparse_input_layer/input_layer/C4_embedding/C4_embedding_weights/strided_slice/stack_2*
_output_shapes
: *
shrink_axis_mask*
Index0*
T0
Ц
Tinput_layer/sparse_input_layer/input_layer/C4_embedding/C4_embedding_weights/stack/0Const*
_output_shapes
: *
dtype0*
value	B :
Ї
Rinput_layer/sparse_input_layer/input_layer/C4_embedding/C4_embedding_weights/stackPackTinput_layer/sparse_input_layer/input_layer/C4_embedding/C4_embedding_weights/stack/0Zinput_layer/sparse_input_layer/input_layer/C4_embedding/C4_embedding_weights/strided_slice*
T0*
_output_shapes
:*
N
ј
Qinput_layer/sparse_input_layer/input_layer/C4_embedding/C4_embedding_weights/TileTileVinput_layer/sparse_input_layer/input_layer/C4_embedding/C4_embedding_weights/Reshape_1Rinput_layer/sparse_input_layer/input_layer/C4_embedding/C4_embedding_weights/stack*0
_output_shapes
:€€€€€€€€€€€€€€€€€€*
T0

ь
Winput_layer/sparse_input_layer/input_layer/C4_embedding/C4_embedding_weights/zeros_like	ZerosLikedinput_layer/sparse_input_layer/input_layer/C4_embedding/C4_embedding_weights/embedding_lookup_sparse*
T0*'
_output_shapes
:€€€€€€€€€
Ъ
Linput_layer/sparse_input_layer/input_layer/C4_embedding/C4_embedding_weightsSelectQinput_layer/sparse_input_layer/input_layer/C4_embedding/C4_embedding_weights/TileWinput_layer/sparse_input_layer/input_layer/C4_embedding/C4_embedding_weights/zeros_likedinput_layer/sparse_input_layer/input_layer/C4_embedding/C4_embedding_weights/embedding_lookup_sparse*
T0*'
_output_shapes
:€€€€€€€€€
д
Sinput_layer/sparse_input_layer/input_layer/C4_embedding/C4_embedding_weights/Cast_1CastSinput_layer/sparse_input_layer/input_layer/C4_embedding/to_sparse_input/dense_shape*

SrcT0	*
_output_shapes
:*

DstT0
§
Zinput_layer/sparse_input_layer/input_layer/C4_embedding/C4_embedding_weights/Slice_1/beginConst*
valueB: *
_output_shapes
:*
dtype0
£
Yinput_layer/sparse_input_layer/input_layer/C4_embedding/C4_embedding_weights/Slice_1/sizeConst*
_output_shapes
:*
valueB:*
dtype0
Ы
Tinput_layer/sparse_input_layer/input_layer/C4_embedding/C4_embedding_weights/Slice_1SliceSinput_layer/sparse_input_layer/input_layer/C4_embedding/C4_embedding_weights/Cast_1Zinput_layer/sparse_input_layer/input_layer/C4_embedding/C4_embedding_weights/Slice_1/beginYinput_layer/sparse_input_layer/input_layer/C4_embedding/C4_embedding_weights/Slice_1/size*
T0*
_output_shapes
:*
Index0
–
Tinput_layer/sparse_input_layer/input_layer/C4_embedding/C4_embedding_weights/Shape_1ShapeLinput_layer/sparse_input_layer/input_layer/C4_embedding/C4_embedding_weights*
_output_shapes
:*
T0
§
Zinput_layer/sparse_input_layer/input_layer/C4_embedding/C4_embedding_weights/Slice_2/beginConst*
valueB:*
dtype0*
_output_shapes
:
ђ
Yinput_layer/sparse_input_layer/input_layer/C4_embedding/C4_embedding_weights/Slice_2/sizeConst*
dtype0*
_output_shapes
:*
valueB:
€€€€€€€€€
Ь
Tinput_layer/sparse_input_layer/input_layer/C4_embedding/C4_embedding_weights/Slice_2SliceTinput_layer/sparse_input_layer/input_layer/C4_embedding/C4_embedding_weights/Shape_1Zinput_layer/sparse_input_layer/input_layer/C4_embedding/C4_embedding_weights/Slice_2/beginYinput_layer/sparse_input_layer/input_layer/C4_embedding/C4_embedding_weights/Slice_2/size*
_output_shapes
:*
Index0*
T0
Ъ
Xinput_layer/sparse_input_layer/input_layer/C4_embedding/C4_embedding_weights/concat/axisConst*
dtype0*
_output_shapes
: *
value	B : 
У
Sinput_layer/sparse_input_layer/input_layer/C4_embedding/C4_embedding_weights/concatConcatV2Tinput_layer/sparse_input_layer/input_layer/C4_embedding/C4_embedding_weights/Slice_1Tinput_layer/sparse_input_layer/input_layer/C4_embedding/C4_embedding_weights/Slice_2Xinput_layer/sparse_input_layer/input_layer/C4_embedding/C4_embedding_weights/concat/axis*
T0*
N*
_output_shapes
:
ґ
Vinput_layer/sparse_input_layer/input_layer/C4_embedding/C4_embedding_weights/Reshape_2ReshapeLinput_layer/sparse_input_layer/input_layer/C4_embedding/C4_embedding_weightsSinput_layer/sparse_input_layer/input_layer/C4_embedding/C4_embedding_weights/concat*
T0*'
_output_shapes
:€€€€€€€€€
√
=input_layer/sparse_input_layer/input_layer/C4_embedding/ShapeShapeVinput_layer/sparse_input_layer/input_layer/C4_embedding/C4_embedding_weights/Reshape_2*
_output_shapes
:*
T0
Х
Kinput_layer/sparse_input_layer/input_layer/C4_embedding/strided_slice/stackConst*
dtype0*
valueB: *
_output_shapes
:
Ч
Minput_layer/sparse_input_layer/input_layer/C4_embedding/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
Ч
Minput_layer/sparse_input_layer/input_layer/C4_embedding/strided_slice/stack_2Const*
valueB:*
_output_shapes
:*
dtype0
≈
Einput_layer/sparse_input_layer/input_layer/C4_embedding/strided_sliceStridedSlice=input_layer/sparse_input_layer/input_layer/C4_embedding/ShapeKinput_layer/sparse_input_layer/input_layer/C4_embedding/strided_slice/stackMinput_layer/sparse_input_layer/input_layer/C4_embedding/strided_slice/stack_1Minput_layer/sparse_input_layer/input_layer/C4_embedding/strided_slice/stack_2*
shrink_axis_mask*
Index0*
_output_shapes
: *
T0
Й
Ginput_layer/sparse_input_layer/input_layer/C4_embedding/Reshape/shape/1Const*
value	B :*
_output_shapes
: *
dtype0
Л
Einput_layer/sparse_input_layer/input_layer/C4_embedding/Reshape/shapePackEinput_layer/sparse_input_layer/input_layer/C4_embedding/strided_sliceGinput_layer/sparse_input_layer/input_layer/C4_embedding/Reshape/shape/1*
T0*
N*
_output_shapes
:
Ы
?input_layer/sparse_input_layer/input_layer/C4_embedding/ReshapeReshapeVinput_layer/sparse_input_layer/input_layer/C4_embedding/C4_embedding_weights/Reshape_2Einput_layer/sparse_input_layer/input_layer/C4_embedding/Reshape/shape*'
_output_shapes
:€€€€€€€€€*
T0
С
Finput_layer/sparse_input_layer/input_layer/C5_embedding/ExpandDims/dimConst*
valueB :
€€€€€€€€€*
dtype0*
_output_shapes
: 
Џ
Binput_layer/sparse_input_layer/input_layer/C5_embedding/ExpandDims
ExpandDimsPlaceholder_17Finput_layer/sparse_input_layer/input_layer/C5_embedding/ExpandDims/dim*'
_output_shapes
:€€€€€€€€€*
T0
Ч
Vinput_layer/sparse_input_layer/input_layer/C5_embedding/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
valueB B *
dtype0
™
Pinput_layer/sparse_input_layer/input_layer/C5_embedding/to_sparse_input/NotEqualNotEqualBinput_layer/sparse_input_layer/input_layer/C5_embedding/ExpandDimsVinput_layer/sparse_input_layer/input_layer/C5_embedding/to_sparse_input/ignore_value/x*'
_output_shapes
:€€€€€€€€€*
T0
”
Oinput_layer/sparse_input_layer/input_layer/C5_embedding/to_sparse_input/indicesWherePinput_layer/sparse_input_layer/input_layer/C5_embedding/to_sparse_input/NotEqual*'
_output_shapes
:€€€€€€€€€
≥
Ninput_layer/sparse_input_layer/input_layer/C5_embedding/to_sparse_input/valuesGatherNdBinput_layer/sparse_input_layer/input_layer/C5_embedding/ExpandDimsOinput_layer/sparse_input_layer/input_layer/C5_embedding/to_sparse_input/indices*
Tindices0	*
Tparams0*#
_output_shapes
:€€€€€€€€€
’
Sinput_layer/sparse_input_layer/input_layer/C5_embedding/to_sparse_input/dense_shapeShapeBinput_layer/sparse_input_layer/input_layer/C5_embedding/ExpandDims*
out_type0	*
T0*
_output_shapes
:
б
>input_layer/sparse_input_layer/input_layer/C5_embedding/lookupStringToHashBucketFastNinput_layer/sparse_input_layer/input_layer/C5_embedding/to_sparse_input/values*#
_output_shapes
:€€€€€€€€€*
num_bucketsРN
Ы
linput_layer/sparse_input_layer/input_layer/C5_embedding/embedding_weights/Initializer/truncated_normal/shapeConst*
dtype0*
valueB"'     *\
_classR
PNloc:@input_layer/sparse_input_layer/input_layer/C5_embedding/embedding_weights*
_output_shapes
:
О
kinput_layer/sparse_input_layer/input_layer/C5_embedding/embedding_weights/Initializer/truncated_normal/meanConst*
dtype0*\
_classR
PNloc:@input_layer/sparse_input_layer/input_layer/C5_embedding/embedding_weights*
_output_shapes
: *
valueB
 *    
Р
minput_layer/sparse_input_layer/input_layer/C5_embedding/embedding_weights/Initializer/truncated_normal/stddevConst*\
_classR
PNloc:@input_layer/sparse_input_layer/input_layer/C5_embedding/embedding_weights*
dtype0*
_output_shapes
: *
valueB
 *  А>
М
vinput_layer/sparse_input_layer/input_layer/C5_embedding/embedding_weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormallinput_layer/sparse_input_layer/input_layer/C5_embedding/embedding_weights/Initializer/truncated_normal/shape*\
_classR
PNloc:@input_layer/sparse_input_layer/input_layer/C5_embedding/embedding_weights*
_output_shapes
:	РN*
dtype0*
T0
а
jinput_layer/sparse_input_layer/input_layer/C5_embedding/embedding_weights/Initializer/truncated_normal/mulMulvinput_layer/sparse_input_layer/input_layer/C5_embedding/embedding_weights/Initializer/truncated_normal/TruncatedNormalminput_layer/sparse_input_layer/input_layer/C5_embedding/embedding_weights/Initializer/truncated_normal/stddev*
_output_shapes
:	РN*\
_classR
PNloc:@input_layer/sparse_input_layer/input_layer/C5_embedding/embedding_weights*
T0
ќ
finput_layer/sparse_input_layer/input_layer/C5_embedding/embedding_weights/Initializer/truncated_normalAddjinput_layer/sparse_input_layer/input_layer/C5_embedding/embedding_weights/Initializer/truncated_normal/mulkinput_layer/sparse_input_layer/input_layer/C5_embedding/embedding_weights/Initializer/truncated_normal/mean*
_output_shapes
:	РN*\
_classR
PNloc:@input_layer/sparse_input_layer/input_layer/C5_embedding/embedding_weights*
T0
щ
Iinput_layer/sparse_input_layer/input_layer/C5_embedding/embedding_weights
VariableV2*
dtype0*\
_classR
PNloc:@input_layer/sparse_input_layer/input_layer/C5_embedding/embedding_weights*
shape:	РN*
_output_shapes
:	РN
Х
Pinput_layer/sparse_input_layer/input_layer/C5_embedding/embedding_weights/AssignAssignIinput_layer/sparse_input_layer/input_layer/C5_embedding/embedding_weightsfinput_layer/sparse_input_layer/input_layer/C5_embedding/embedding_weights/Initializer/truncated_normal*\
_classR
PNloc:@input_layer/sparse_input_layer/input_layer/C5_embedding/embedding_weights*
T0*
_output_shapes
:	РN
≠
Ninput_layer/sparse_input_layer/input_layer/C5_embedding/embedding_weights/readIdentityIinput_layer/sparse_input_layer/input_layer/C5_embedding/embedding_weights*
_output_shapes
:	РN*\
_classR
PNloc:@input_layer/sparse_input_layer/input_layer/C5_embedding/embedding_weights*
T0
Ґ
Xinput_layer/sparse_input_layer/input_layer/C5_embedding/C5_embedding_weights/Slice/beginConst*
_output_shapes
:*
dtype0*
valueB: 
°
Winput_layer/sparse_input_layer/input_layer/C5_embedding/C5_embedding_weights/Slice/sizeConst*
dtype0*
valueB:*
_output_shapes
:
Х
Rinput_layer/sparse_input_layer/input_layer/C5_embedding/C5_embedding_weights/SliceSliceSinput_layer/sparse_input_layer/input_layer/C5_embedding/to_sparse_input/dense_shapeXinput_layer/sparse_input_layer/input_layer/C5_embedding/C5_embedding_weights/Slice/beginWinput_layer/sparse_input_layer/input_layer/C5_embedding/C5_embedding_weights/Slice/size*
_output_shapes
:*
Index0*
T0	
Ь
Rinput_layer/sparse_input_layer/input_layer/C5_embedding/C5_embedding_weights/ConstConst*
valueB: *
dtype0*
_output_shapes
:
Ґ
Qinput_layer/sparse_input_layer/input_layer/C5_embedding/C5_embedding_weights/ProdProdRinput_layer/sparse_input_layer/input_layer/C5_embedding/C5_embedding_weights/SliceRinput_layer/sparse_input_layer/input_layer/C5_embedding/C5_embedding_weights/Const*
_output_shapes
: *
T0	
Я
]input_layer/sparse_input_layer/input_layer/C5_embedding/C5_embedding_weights/GatherV2/indicesConst*
value	B :*
dtype0*
_output_shapes
: 
Ь
Zinput_layer/sparse_input_layer/input_layer/C5_embedding/C5_embedding_weights/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 
µ
Uinput_layer/sparse_input_layer/input_layer/C5_embedding/C5_embedding_weights/GatherV2GatherV2Sinput_layer/sparse_input_layer/input_layer/C5_embedding/to_sparse_input/dense_shape]input_layer/sparse_input_layer/input_layer/C5_embedding/C5_embedding_weights/GatherV2/indicesZinput_layer/sparse_input_layer/input_layer/C5_embedding/C5_embedding_weights/GatherV2/axis*
Taxis0*
Tindices0*
_output_shapes
: *
Tparams0	
≥
Sinput_layer/sparse_input_layer/input_layer/C5_embedding/C5_embedding_weights/Cast/xPackQinput_layer/sparse_input_layer/input_layer/C5_embedding/C5_embedding_weights/ProdUinput_layer/sparse_input_layer/input_layer/C5_embedding/C5_embedding_weights/GatherV2*
T0	*
N*
_output_shapes
:
Х
Zinput_layer/sparse_input_layer/input_layer/C5_embedding/C5_embedding_weights/SparseReshapeSparseReshapeOinput_layer/sparse_input_layer/input_layer/C5_embedding/to_sparse_input/indicesSinput_layer/sparse_input_layer/input_layer/C5_embedding/to_sparse_input/dense_shapeSinput_layer/sparse_input_layer/input_layer/C5_embedding/C5_embedding_weights/Cast/x*-
_output_shapes
:€€€€€€€€€:
Ё
cinput_layer/sparse_input_layer/input_layer/C5_embedding/C5_embedding_weights/SparseReshape/IdentityIdentity>input_layer/sparse_input_layer/input_layer/C5_embedding/lookup*#
_output_shapes
:€€€€€€€€€*
T0	
Э
[input_layer/sparse_input_layer/input_layer/C5_embedding/C5_embedding_weights/GreaterEqual/yConst*
_output_shapes
: *
value	B	 R *
dtype0	
ў
Yinput_layer/sparse_input_layer/input_layer/C5_embedding/C5_embedding_weights/GreaterEqualGreaterEqualcinput_layer/sparse_input_layer/input_layer/C5_embedding/C5_embedding_weights/SparseReshape/Identity[input_layer/sparse_input_layer/input_layer/C5_embedding/C5_embedding_weights/GreaterEqual/y*
T0	*#
_output_shapes
:€€€€€€€€€
я
Rinput_layer/sparse_input_layer/input_layer/C5_embedding/C5_embedding_weights/WhereWhereYinput_layer/sparse_input_layer/input_layer/C5_embedding/C5_embedding_weights/GreaterEqual*'
_output_shapes
:€€€€€€€€€
≠
Zinput_layer/sparse_input_layer/input_layer/C5_embedding/C5_embedding_weights/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€
љ
Tinput_layer/sparse_input_layer/input_layer/C5_embedding/C5_embedding_weights/ReshapeReshapeRinput_layer/sparse_input_layer/input_layer/C5_embedding/C5_embedding_weights/WhereZinput_layer/sparse_input_layer/input_layer/C5_embedding/C5_embedding_weights/Reshape/shape*
T0	*#
_output_shapes
:€€€€€€€€€
Ю
\input_layer/sparse_input_layer/input_layer/C5_embedding/C5_embedding_weights/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
»
Winput_layer/sparse_input_layer/input_layer/C5_embedding/C5_embedding_weights/GatherV2_1GatherV2Zinput_layer/sparse_input_layer/input_layer/C5_embedding/C5_embedding_weights/SparseReshapeTinput_layer/sparse_input_layer/input_layer/C5_embedding/C5_embedding_weights/Reshape\input_layer/sparse_input_layer/input_layer/C5_embedding/C5_embedding_weights/GatherV2_1/axis*
Tindices0	*
Taxis0*
Tparams0	*'
_output_shapes
:€€€€€€€€€
Ю
\input_layer/sparse_input_layer/input_layer/C5_embedding/C5_embedding_weights/GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
value	B : 
Ќ
Winput_layer/sparse_input_layer/input_layer/C5_embedding/C5_embedding_weights/GatherV2_2GatherV2cinput_layer/sparse_input_layer/input_layer/C5_embedding/C5_embedding_weights/SparseReshape/IdentityTinput_layer/sparse_input_layer/input_layer/C5_embedding/C5_embedding_weights/Reshape\input_layer/sparse_input_layer/input_layer/C5_embedding/C5_embedding_weights/GatherV2_2/axis*
Tindices0	*#
_output_shapes
:€€€€€€€€€*
Tparams0	*
Taxis0
д
Uinput_layer/sparse_input_layer/input_layer/C5_embedding/C5_embedding_weights/IdentityIdentity\input_layer/sparse_input_layer/input_layer/C5_embedding/C5_embedding_weights/SparseReshape:1*
_output_shapes
:*
T0	
®
finput_layer/sparse_input_layer/input_layer/C5_embedding/C5_embedding_weights/SparseFillEmptyRows/ConstConst*
value	B	 R *
dtype0	*
_output_shapes
: 
џ
tinput_layer/sparse_input_layer/input_layer/C5_embedding/C5_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowsSparseFillEmptyRowsWinput_layer/sparse_input_layer/input_layer/C5_embedding/C5_embedding_weights/GatherV2_1Winput_layer/sparse_input_layer/input_layer/C5_embedding/C5_embedding_weights/GatherV2_2Uinput_layer/sparse_input_layer/input_layer/C5_embedding/C5_embedding_weights/Identityfinput_layer/sparse_input_layer/input_layer/C5_embedding/C5_embedding_weights/SparseFillEmptyRows/Const*
T0	*T
_output_shapesB
@:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€
…
xinput_layer/sparse_input_layer/input_layer/C5_embedding/C5_embedding_weights/embedding_lookup_sparse/strided_slice/stackConst*
dtype0*
valueB"        *
_output_shapes
:
Ћ
zinput_layer/sparse_input_layer/input_layer/C5_embedding/C5_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB"       
Ћ
zinput_layer/sparse_input_layer/input_layer/C5_embedding/C5_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2Const*
valueB"      *
_output_shapes
:*
dtype0
я
rinput_layer/sparse_input_layer/input_layer/C5_embedding/C5_embedding_weights/embedding_lookup_sparse/strided_sliceStridedSlicetinput_layer/sparse_input_layer/input_layer/C5_embedding/C5_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowsxinput_layer/sparse_input_layer/input_layer/C5_embedding/C5_embedding_weights/embedding_lookup_sparse/strided_slice/stackzinput_layer/sparse_input_layer/input_layer/C5_embedding/C5_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1zinput_layer/sparse_input_layer/input_layer/C5_embedding/C5_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2*
Index0*#
_output_shapes
:€€€€€€€€€*
T0	*
end_mask*
shrink_axis_mask*

begin_mask
Ґ
iinput_layer/sparse_input_layer/input_layer/C5_embedding/C5_embedding_weights/embedding_lookup_sparse/CastCastrinput_layer/sparse_input_layer/input_layer/C5_embedding/C5_embedding_weights/embedding_lookup_sparse/strided_slice*

DstT0*#
_output_shapes
:€€€€€€€€€*

SrcT0	
™
kinput_layer/sparse_input_layer/input_layer/C5_embedding/C5_embedding_weights/embedding_lookup_sparse/UniqueUniquevinput_layer/sparse_input_layer/input_layer/C5_embedding/C5_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€*
T0	
Ъ
zinput_layer/sparse_input_layer/input_layer/C5_embedding/C5_embedding_weights/embedding_lookup_sparse/embedding_lookup/axisConst*
value	B : *
_output_shapes
: *
dtype0*\
_classR
PNloc:@input_layer/sparse_input_layer/input_layer/C5_embedding/embedding_weights
н
uinput_layer/sparse_input_layer/input_layer/C5_embedding/C5_embedding_weights/embedding_lookup_sparse/embedding_lookupGatherV2Ninput_layer/sparse_input_layer/input_layer/C5_embedding/embedding_weights/readkinput_layer/sparse_input_layer/input_layer/C5_embedding/C5_embedding_weights/embedding_lookup_sparse/Uniquezinput_layer/sparse_input_layer/input_layer/C5_embedding/C5_embedding_weights/embedding_lookup_sparse/embedding_lookup/axis*
Taxis0*\
_classR
PNloc:@input_layer/sparse_input_layer/input_layer/C5_embedding/embedding_weights*'
_output_shapes
:€€€€€€€€€*
Tindices0	*
Tparams0
≥
~input_layer/sparse_input_layer/input_layer/C5_embedding/C5_embedding_weights/embedding_lookup_sparse/embedding_lookup/IdentityIdentityuinput_layer/sparse_input_layer/input_layer/C5_embedding/C5_embedding_weights/embedding_lookup_sparse/embedding_lookup*'
_output_shapes
:€€€€€€€€€*
T0
Е
dinput_layer/sparse_input_layer/input_layer/C5_embedding/C5_embedding_weights/embedding_lookup_sparseSparseSegmentMean~input_layer/sparse_input_layer/input_layer/C5_embedding/C5_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identityminput_layer/sparse_input_layer/input_layer/C5_embedding/C5_embedding_weights/embedding_lookup_sparse/Unique:1iinput_layer/sparse_input_layer/input_layer/C5_embedding/C5_embedding_weights/embedding_lookup_sparse/Cast*'
_output_shapes
:€€€€€€€€€*
T0
≠
\input_layer/sparse_input_layer/input_layer/C5_embedding/C5_embedding_weights/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   
й
Vinput_layer/sparse_input_layer/input_layer/C5_embedding/C5_embedding_weights/Reshape_1Reshapevinput_layer/sparse_input_layer/input_layer/C5_embedding/C5_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:2\input_layer/sparse_input_layer/input_layer/C5_embedding/C5_embedding_weights/Reshape_1/shape*
T0
*'
_output_shapes
:€€€€€€€€€
ж
Rinput_layer/sparse_input_layer/input_layer/C5_embedding/C5_embedding_weights/ShapeShapedinput_layer/sparse_input_layer/input_layer/C5_embedding/C5_embedding_weights/embedding_lookup_sparse*
_output_shapes
:*
T0
™
`input_layer/sparse_input_layer/input_layer/C5_embedding/C5_embedding_weights/strided_slice/stackConst*
valueB:*
dtype0*
_output_shapes
:
ђ
binput_layer/sparse_input_layer/input_layer/C5_embedding/C5_embedding_weights/strided_slice/stack_1Const*
dtype0*
valueB:*
_output_shapes
:
ђ
binput_layer/sparse_input_layer/input_layer/C5_embedding/C5_embedding_weights/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Ѓ
Zinput_layer/sparse_input_layer/input_layer/C5_embedding/C5_embedding_weights/strided_sliceStridedSliceRinput_layer/sparse_input_layer/input_layer/C5_embedding/C5_embedding_weights/Shape`input_layer/sparse_input_layer/input_layer/C5_embedding/C5_embedding_weights/strided_slice/stackbinput_layer/sparse_input_layer/input_layer/C5_embedding/C5_embedding_weights/strided_slice/stack_1binput_layer/sparse_input_layer/input_layer/C5_embedding/C5_embedding_weights/strided_slice/stack_2*
shrink_axis_mask*
Index0*
T0*
_output_shapes
: 
Ц
Tinput_layer/sparse_input_layer/input_layer/C5_embedding/C5_embedding_weights/stack/0Const*
value	B :*
dtype0*
_output_shapes
: 
Ї
Rinput_layer/sparse_input_layer/input_layer/C5_embedding/C5_embedding_weights/stackPackTinput_layer/sparse_input_layer/input_layer/C5_embedding/C5_embedding_weights/stack/0Zinput_layer/sparse_input_layer/input_layer/C5_embedding/C5_embedding_weights/strided_slice*
N*
T0*
_output_shapes
:
ј
Qinput_layer/sparse_input_layer/input_layer/C5_embedding/C5_embedding_weights/TileTileVinput_layer/sparse_input_layer/input_layer/C5_embedding/C5_embedding_weights/Reshape_1Rinput_layer/sparse_input_layer/input_layer/C5_embedding/C5_embedding_weights/stack*0
_output_shapes
:€€€€€€€€€€€€€€€€€€*
T0

ь
Winput_layer/sparse_input_layer/input_layer/C5_embedding/C5_embedding_weights/zeros_like	ZerosLikedinput_layer/sparse_input_layer/input_layer/C5_embedding/C5_embedding_weights/embedding_lookup_sparse*'
_output_shapes
:€€€€€€€€€*
T0
Ъ
Linput_layer/sparse_input_layer/input_layer/C5_embedding/C5_embedding_weightsSelectQinput_layer/sparse_input_layer/input_layer/C5_embedding/C5_embedding_weights/TileWinput_layer/sparse_input_layer/input_layer/C5_embedding/C5_embedding_weights/zeros_likedinput_layer/sparse_input_layer/input_layer/C5_embedding/C5_embedding_weights/embedding_lookup_sparse*'
_output_shapes
:€€€€€€€€€*
T0
д
Sinput_layer/sparse_input_layer/input_layer/C5_embedding/C5_embedding_weights/Cast_1CastSinput_layer/sparse_input_layer/input_layer/C5_embedding/to_sparse_input/dense_shape*

SrcT0	*
_output_shapes
:*

DstT0
§
Zinput_layer/sparse_input_layer/input_layer/C5_embedding/C5_embedding_weights/Slice_1/beginConst*
dtype0*
_output_shapes
:*
valueB: 
£
Yinput_layer/sparse_input_layer/input_layer/C5_embedding/C5_embedding_weights/Slice_1/sizeConst*
valueB:*
_output_shapes
:*
dtype0
Ы
Tinput_layer/sparse_input_layer/input_layer/C5_embedding/C5_embedding_weights/Slice_1SliceSinput_layer/sparse_input_layer/input_layer/C5_embedding/C5_embedding_weights/Cast_1Zinput_layer/sparse_input_layer/input_layer/C5_embedding/C5_embedding_weights/Slice_1/beginYinput_layer/sparse_input_layer/input_layer/C5_embedding/C5_embedding_weights/Slice_1/size*
T0*
_output_shapes
:*
Index0
–
Tinput_layer/sparse_input_layer/input_layer/C5_embedding/C5_embedding_weights/Shape_1ShapeLinput_layer/sparse_input_layer/input_layer/C5_embedding/C5_embedding_weights*
T0*
_output_shapes
:
§
Zinput_layer/sparse_input_layer/input_layer/C5_embedding/C5_embedding_weights/Slice_2/beginConst*
valueB:*
dtype0*
_output_shapes
:
ђ
Yinput_layer/sparse_input_layer/input_layer/C5_embedding/C5_embedding_weights/Slice_2/sizeConst*
valueB:
€€€€€€€€€*
_output_shapes
:*
dtype0
Ь
Tinput_layer/sparse_input_layer/input_layer/C5_embedding/C5_embedding_weights/Slice_2SliceTinput_layer/sparse_input_layer/input_layer/C5_embedding/C5_embedding_weights/Shape_1Zinput_layer/sparse_input_layer/input_layer/C5_embedding/C5_embedding_weights/Slice_2/beginYinput_layer/sparse_input_layer/input_layer/C5_embedding/C5_embedding_weights/Slice_2/size*
Index0*
_output_shapes
:*
T0
Ъ
Xinput_layer/sparse_input_layer/input_layer/C5_embedding/C5_embedding_weights/concat/axisConst*
_output_shapes
: *
value	B : *
dtype0
У
Sinput_layer/sparse_input_layer/input_layer/C5_embedding/C5_embedding_weights/concatConcatV2Tinput_layer/sparse_input_layer/input_layer/C5_embedding/C5_embedding_weights/Slice_1Tinput_layer/sparse_input_layer/input_layer/C5_embedding/C5_embedding_weights/Slice_2Xinput_layer/sparse_input_layer/input_layer/C5_embedding/C5_embedding_weights/concat/axis*
N*
_output_shapes
:*
T0
ґ
Vinput_layer/sparse_input_layer/input_layer/C5_embedding/C5_embedding_weights/Reshape_2ReshapeLinput_layer/sparse_input_layer/input_layer/C5_embedding/C5_embedding_weightsSinput_layer/sparse_input_layer/input_layer/C5_embedding/C5_embedding_weights/concat*
T0*'
_output_shapes
:€€€€€€€€€
√
=input_layer/sparse_input_layer/input_layer/C5_embedding/ShapeShapeVinput_layer/sparse_input_layer/input_layer/C5_embedding/C5_embedding_weights/Reshape_2*
T0*
_output_shapes
:
Х
Kinput_layer/sparse_input_layer/input_layer/C5_embedding/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Ч
Minput_layer/sparse_input_layer/input_layer/C5_embedding/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Ч
Minput_layer/sparse_input_layer/input_layer/C5_embedding/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
≈
Einput_layer/sparse_input_layer/input_layer/C5_embedding/strided_sliceStridedSlice=input_layer/sparse_input_layer/input_layer/C5_embedding/ShapeKinput_layer/sparse_input_layer/input_layer/C5_embedding/strided_slice/stackMinput_layer/sparse_input_layer/input_layer/C5_embedding/strided_slice/stack_1Minput_layer/sparse_input_layer/input_layer/C5_embedding/strided_slice/stack_2*
Index0*
_output_shapes
: *
shrink_axis_mask*
T0
Й
Ginput_layer/sparse_input_layer/input_layer/C5_embedding/Reshape/shape/1Const*
dtype0*
_output_shapes
: *
value	B :
Л
Einput_layer/sparse_input_layer/input_layer/C5_embedding/Reshape/shapePackEinput_layer/sparse_input_layer/input_layer/C5_embedding/strided_sliceGinput_layer/sparse_input_layer/input_layer/C5_embedding/Reshape/shape/1*
T0*
_output_shapes
:*
N
Ы
?input_layer/sparse_input_layer/input_layer/C5_embedding/ReshapeReshapeVinput_layer/sparse_input_layer/input_layer/C5_embedding/C5_embedding_weights/Reshape_2Einput_layer/sparse_input_layer/input_layer/C5_embedding/Reshape/shape*'
_output_shapes
:€€€€€€€€€*
T0
С
Finput_layer/sparse_input_layer/input_layer/C6_embedding/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€
Џ
Binput_layer/sparse_input_layer/input_layer/C6_embedding/ExpandDims
ExpandDimsPlaceholder_18Finput_layer/sparse_input_layer/input_layer/C6_embedding/ExpandDims/dim*
T0*'
_output_shapes
:€€€€€€€€€
Ч
Vinput_layer/sparse_input_layer/input_layer/C6_embedding/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB B 
™
Pinput_layer/sparse_input_layer/input_layer/C6_embedding/to_sparse_input/NotEqualNotEqualBinput_layer/sparse_input_layer/input_layer/C6_embedding/ExpandDimsVinput_layer/sparse_input_layer/input_layer/C6_embedding/to_sparse_input/ignore_value/x*
T0*'
_output_shapes
:€€€€€€€€€
”
Oinput_layer/sparse_input_layer/input_layer/C6_embedding/to_sparse_input/indicesWherePinput_layer/sparse_input_layer/input_layer/C6_embedding/to_sparse_input/NotEqual*'
_output_shapes
:€€€€€€€€€
≥
Ninput_layer/sparse_input_layer/input_layer/C6_embedding/to_sparse_input/valuesGatherNdBinput_layer/sparse_input_layer/input_layer/C6_embedding/ExpandDimsOinput_layer/sparse_input_layer/input_layer/C6_embedding/to_sparse_input/indices*
Tparams0*
Tindices0	*#
_output_shapes
:€€€€€€€€€
’
Sinput_layer/sparse_input_layer/input_layer/C6_embedding/to_sparse_input/dense_shapeShapeBinput_layer/sparse_input_layer/input_layer/C6_embedding/ExpandDims*
out_type0	*
T0*
_output_shapes
:
б
>input_layer/sparse_input_layer/input_layer/C6_embedding/lookupStringToHashBucketFastNinput_layer/sparse_input_layer/input_layer/C6_embedding/to_sparse_input/values*
num_bucketsРN*#
_output_shapes
:€€€€€€€€€
Ы
linput_layer/sparse_input_layer/input_layer/C6_embedding/embedding_weights/Initializer/truncated_normal/shapeConst*\
_classR
PNloc:@input_layer/sparse_input_layer/input_layer/C6_embedding/embedding_weights*
valueB"'     *
dtype0*
_output_shapes
:
О
kinput_layer/sparse_input_layer/input_layer/C6_embedding/embedding_weights/Initializer/truncated_normal/meanConst*
valueB
 *    *\
_classR
PNloc:@input_layer/sparse_input_layer/input_layer/C6_embedding/embedding_weights*
_output_shapes
: *
dtype0
Р
minput_layer/sparse_input_layer/input_layer/C6_embedding/embedding_weights/Initializer/truncated_normal/stddevConst*
_output_shapes
: *\
_classR
PNloc:@input_layer/sparse_input_layer/input_layer/C6_embedding/embedding_weights*
valueB
 *  А>*
dtype0
М
vinput_layer/sparse_input_layer/input_layer/C6_embedding/embedding_weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormallinput_layer/sparse_input_layer/input_layer/C6_embedding/embedding_weights/Initializer/truncated_normal/shape*
_output_shapes
:	РN*
dtype0*
T0*\
_classR
PNloc:@input_layer/sparse_input_layer/input_layer/C6_embedding/embedding_weights
а
jinput_layer/sparse_input_layer/input_layer/C6_embedding/embedding_weights/Initializer/truncated_normal/mulMulvinput_layer/sparse_input_layer/input_layer/C6_embedding/embedding_weights/Initializer/truncated_normal/TruncatedNormalminput_layer/sparse_input_layer/input_layer/C6_embedding/embedding_weights/Initializer/truncated_normal/stddev*
_output_shapes
:	РN*
T0*\
_classR
PNloc:@input_layer/sparse_input_layer/input_layer/C6_embedding/embedding_weights
ќ
finput_layer/sparse_input_layer/input_layer/C6_embedding/embedding_weights/Initializer/truncated_normalAddjinput_layer/sparse_input_layer/input_layer/C6_embedding/embedding_weights/Initializer/truncated_normal/mulkinput_layer/sparse_input_layer/input_layer/C6_embedding/embedding_weights/Initializer/truncated_normal/mean*
_output_shapes
:	РN*\
_classR
PNloc:@input_layer/sparse_input_layer/input_layer/C6_embedding/embedding_weights*
T0
щ
Iinput_layer/sparse_input_layer/input_layer/C6_embedding/embedding_weights
VariableV2*
_output_shapes
:	РN*
dtype0*
shape:	РN*\
_classR
PNloc:@input_layer/sparse_input_layer/input_layer/C6_embedding/embedding_weights
Х
Pinput_layer/sparse_input_layer/input_layer/C6_embedding/embedding_weights/AssignAssignIinput_layer/sparse_input_layer/input_layer/C6_embedding/embedding_weightsfinput_layer/sparse_input_layer/input_layer/C6_embedding/embedding_weights/Initializer/truncated_normal*\
_classR
PNloc:@input_layer/sparse_input_layer/input_layer/C6_embedding/embedding_weights*
T0*
_output_shapes
:	РN
≠
Ninput_layer/sparse_input_layer/input_layer/C6_embedding/embedding_weights/readIdentityIinput_layer/sparse_input_layer/input_layer/C6_embedding/embedding_weights*
_output_shapes
:	РN*
T0*\
_classR
PNloc:@input_layer/sparse_input_layer/input_layer/C6_embedding/embedding_weights
Ґ
Xinput_layer/sparse_input_layer/input_layer/C6_embedding/C6_embedding_weights/Slice/beginConst*
dtype0*
_output_shapes
:*
valueB: 
°
Winput_layer/sparse_input_layer/input_layer/C6_embedding/C6_embedding_weights/Slice/sizeConst*
valueB:*
dtype0*
_output_shapes
:
Х
Rinput_layer/sparse_input_layer/input_layer/C6_embedding/C6_embedding_weights/SliceSliceSinput_layer/sparse_input_layer/input_layer/C6_embedding/to_sparse_input/dense_shapeXinput_layer/sparse_input_layer/input_layer/C6_embedding/C6_embedding_weights/Slice/beginWinput_layer/sparse_input_layer/input_layer/C6_embedding/C6_embedding_weights/Slice/size*
_output_shapes
:*
T0	*
Index0
Ь
Rinput_layer/sparse_input_layer/input_layer/C6_embedding/C6_embedding_weights/ConstConst*
dtype0*
_output_shapes
:*
valueB: 
Ґ
Qinput_layer/sparse_input_layer/input_layer/C6_embedding/C6_embedding_weights/ProdProdRinput_layer/sparse_input_layer/input_layer/C6_embedding/C6_embedding_weights/SliceRinput_layer/sparse_input_layer/input_layer/C6_embedding/C6_embedding_weights/Const*
T0	*
_output_shapes
: 
Я
]input_layer/sparse_input_layer/input_layer/C6_embedding/C6_embedding_weights/GatherV2/indicesConst*
_output_shapes
: *
value	B :*
dtype0
Ь
Zinput_layer/sparse_input_layer/input_layer/C6_embedding/C6_embedding_weights/GatherV2/axisConst*
dtype0*
value	B : *
_output_shapes
: 
µ
Uinput_layer/sparse_input_layer/input_layer/C6_embedding/C6_embedding_weights/GatherV2GatherV2Sinput_layer/sparse_input_layer/input_layer/C6_embedding/to_sparse_input/dense_shape]input_layer/sparse_input_layer/input_layer/C6_embedding/C6_embedding_weights/GatherV2/indicesZinput_layer/sparse_input_layer/input_layer/C6_embedding/C6_embedding_weights/GatherV2/axis*
Tindices0*
Taxis0*
_output_shapes
: *
Tparams0	
≥
Sinput_layer/sparse_input_layer/input_layer/C6_embedding/C6_embedding_weights/Cast/xPackQinput_layer/sparse_input_layer/input_layer/C6_embedding/C6_embedding_weights/ProdUinput_layer/sparse_input_layer/input_layer/C6_embedding/C6_embedding_weights/GatherV2*
T0	*
N*
_output_shapes
:
Х
Zinput_layer/sparse_input_layer/input_layer/C6_embedding/C6_embedding_weights/SparseReshapeSparseReshapeOinput_layer/sparse_input_layer/input_layer/C6_embedding/to_sparse_input/indicesSinput_layer/sparse_input_layer/input_layer/C6_embedding/to_sparse_input/dense_shapeSinput_layer/sparse_input_layer/input_layer/C6_embedding/C6_embedding_weights/Cast/x*-
_output_shapes
:€€€€€€€€€:
Ё
cinput_layer/sparse_input_layer/input_layer/C6_embedding/C6_embedding_weights/SparseReshape/IdentityIdentity>input_layer/sparse_input_layer/input_layer/C6_embedding/lookup*
T0	*#
_output_shapes
:€€€€€€€€€
Э
[input_layer/sparse_input_layer/input_layer/C6_embedding/C6_embedding_weights/GreaterEqual/yConst*
_output_shapes
: *
dtype0	*
value	B	 R 
ў
Yinput_layer/sparse_input_layer/input_layer/C6_embedding/C6_embedding_weights/GreaterEqualGreaterEqualcinput_layer/sparse_input_layer/input_layer/C6_embedding/C6_embedding_weights/SparseReshape/Identity[input_layer/sparse_input_layer/input_layer/C6_embedding/C6_embedding_weights/GreaterEqual/y*
T0	*#
_output_shapes
:€€€€€€€€€
я
Rinput_layer/sparse_input_layer/input_layer/C6_embedding/C6_embedding_weights/WhereWhereYinput_layer/sparse_input_layer/input_layer/C6_embedding/C6_embedding_weights/GreaterEqual*'
_output_shapes
:€€€€€€€€€
≠
Zinput_layer/sparse_input_layer/input_layer/C6_embedding/C6_embedding_weights/Reshape/shapeConst*
_output_shapes
:*
valueB:
€€€€€€€€€*
dtype0
љ
Tinput_layer/sparse_input_layer/input_layer/C6_embedding/C6_embedding_weights/ReshapeReshapeRinput_layer/sparse_input_layer/input_layer/C6_embedding/C6_embedding_weights/WhereZinput_layer/sparse_input_layer/input_layer/C6_embedding/C6_embedding_weights/Reshape/shape*#
_output_shapes
:€€€€€€€€€*
T0	
Ю
\input_layer/sparse_input_layer/input_layer/C6_embedding/C6_embedding_weights/GatherV2_1/axisConst*
dtype0*
value	B : *
_output_shapes
: 
»
Winput_layer/sparse_input_layer/input_layer/C6_embedding/C6_embedding_weights/GatherV2_1GatherV2Zinput_layer/sparse_input_layer/input_layer/C6_embedding/C6_embedding_weights/SparseReshapeTinput_layer/sparse_input_layer/input_layer/C6_embedding/C6_embedding_weights/Reshape\input_layer/sparse_input_layer/input_layer/C6_embedding/C6_embedding_weights/GatherV2_1/axis*
Taxis0*
Tindices0	*
Tparams0	*'
_output_shapes
:€€€€€€€€€
Ю
\input_layer/sparse_input_layer/input_layer/C6_embedding/C6_embedding_weights/GatherV2_2/axisConst*
value	B : *
dtype0*
_output_shapes
: 
Ќ
Winput_layer/sparse_input_layer/input_layer/C6_embedding/C6_embedding_weights/GatherV2_2GatherV2cinput_layer/sparse_input_layer/input_layer/C6_embedding/C6_embedding_weights/SparseReshape/IdentityTinput_layer/sparse_input_layer/input_layer/C6_embedding/C6_embedding_weights/Reshape\input_layer/sparse_input_layer/input_layer/C6_embedding/C6_embedding_weights/GatherV2_2/axis*
Taxis0*#
_output_shapes
:€€€€€€€€€*
Tindices0	*
Tparams0	
д
Uinput_layer/sparse_input_layer/input_layer/C6_embedding/C6_embedding_weights/IdentityIdentity\input_layer/sparse_input_layer/input_layer/C6_embedding/C6_embedding_weights/SparseReshape:1*
_output_shapes
:*
T0	
®
finput_layer/sparse_input_layer/input_layer/C6_embedding/C6_embedding_weights/SparseFillEmptyRows/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R 
џ
tinput_layer/sparse_input_layer/input_layer/C6_embedding/C6_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowsSparseFillEmptyRowsWinput_layer/sparse_input_layer/input_layer/C6_embedding/C6_embedding_weights/GatherV2_1Winput_layer/sparse_input_layer/input_layer/C6_embedding/C6_embedding_weights/GatherV2_2Uinput_layer/sparse_input_layer/input_layer/C6_embedding/C6_embedding_weights/Identityfinput_layer/sparse_input_layer/input_layer/C6_embedding/C6_embedding_weights/SparseFillEmptyRows/Const*T
_output_shapesB
@:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€*
T0	
…
xinput_layer/sparse_input_layer/input_layer/C6_embedding/C6_embedding_weights/embedding_lookup_sparse/strided_slice/stackConst*
valueB"        *
_output_shapes
:*
dtype0
Ћ
zinput_layer/sparse_input_layer/input_layer/C6_embedding/C6_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1Const*
dtype0*
valueB"       *
_output_shapes
:
Ћ
zinput_layer/sparse_input_layer/input_layer/C6_embedding/C6_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB"      
я
rinput_layer/sparse_input_layer/input_layer/C6_embedding/C6_embedding_weights/embedding_lookup_sparse/strided_sliceStridedSlicetinput_layer/sparse_input_layer/input_layer/C6_embedding/C6_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowsxinput_layer/sparse_input_layer/input_layer/C6_embedding/C6_embedding_weights/embedding_lookup_sparse/strided_slice/stackzinput_layer/sparse_input_layer/input_layer/C6_embedding/C6_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1zinput_layer/sparse_input_layer/input_layer/C6_embedding/C6_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2*
Index0*
T0	*#
_output_shapes
:€€€€€€€€€*
shrink_axis_mask*
end_mask*

begin_mask
Ґ
iinput_layer/sparse_input_layer/input_layer/C6_embedding/C6_embedding_weights/embedding_lookup_sparse/CastCastrinput_layer/sparse_input_layer/input_layer/C6_embedding/C6_embedding_weights/embedding_lookup_sparse/strided_slice*

SrcT0	*

DstT0*#
_output_shapes
:€€€€€€€€€
™
kinput_layer/sparse_input_layer/input_layer/C6_embedding/C6_embedding_weights/embedding_lookup_sparse/UniqueUniquevinput_layer/sparse_input_layer/input_layer/C6_embedding/C6_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€*
T0	
Ъ
zinput_layer/sparse_input_layer/input_layer/C6_embedding/C6_embedding_weights/embedding_lookup_sparse/embedding_lookup/axisConst*\
_classR
PNloc:@input_layer/sparse_input_layer/input_layer/C6_embedding/embedding_weights*
_output_shapes
: *
dtype0*
value	B : 
н
uinput_layer/sparse_input_layer/input_layer/C6_embedding/C6_embedding_weights/embedding_lookup_sparse/embedding_lookupGatherV2Ninput_layer/sparse_input_layer/input_layer/C6_embedding/embedding_weights/readkinput_layer/sparse_input_layer/input_layer/C6_embedding/C6_embedding_weights/embedding_lookup_sparse/Uniquezinput_layer/sparse_input_layer/input_layer/C6_embedding/C6_embedding_weights/embedding_lookup_sparse/embedding_lookup/axis*\
_classR
PNloc:@input_layer/sparse_input_layer/input_layer/C6_embedding/embedding_weights*
Tindices0	*
Taxis0*'
_output_shapes
:€€€€€€€€€*
Tparams0
≥
~input_layer/sparse_input_layer/input_layer/C6_embedding/C6_embedding_weights/embedding_lookup_sparse/embedding_lookup/IdentityIdentityuinput_layer/sparse_input_layer/input_layer/C6_embedding/C6_embedding_weights/embedding_lookup_sparse/embedding_lookup*
T0*'
_output_shapes
:€€€€€€€€€
Е
dinput_layer/sparse_input_layer/input_layer/C6_embedding/C6_embedding_weights/embedding_lookup_sparseSparseSegmentMean~input_layer/sparse_input_layer/input_layer/C6_embedding/C6_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identityminput_layer/sparse_input_layer/input_layer/C6_embedding/C6_embedding_weights/embedding_lookup_sparse/Unique:1iinput_layer/sparse_input_layer/input_layer/C6_embedding/C6_embedding_weights/embedding_lookup_sparse/Cast*
T0*'
_output_shapes
:€€€€€€€€€
≠
\input_layer/sparse_input_layer/input_layer/C6_embedding/C6_embedding_weights/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   
й
Vinput_layer/sparse_input_layer/input_layer/C6_embedding/C6_embedding_weights/Reshape_1Reshapevinput_layer/sparse_input_layer/input_layer/C6_embedding/C6_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:2\input_layer/sparse_input_layer/input_layer/C6_embedding/C6_embedding_weights/Reshape_1/shape*'
_output_shapes
:€€€€€€€€€*
T0

ж
Rinput_layer/sparse_input_layer/input_layer/C6_embedding/C6_embedding_weights/ShapeShapedinput_layer/sparse_input_layer/input_layer/C6_embedding/C6_embedding_weights/embedding_lookup_sparse*
_output_shapes
:*
T0
™
`input_layer/sparse_input_layer/input_layer/C6_embedding/C6_embedding_weights/strided_slice/stackConst*
dtype0*
valueB:*
_output_shapes
:
ђ
binput_layer/sparse_input_layer/input_layer/C6_embedding/C6_embedding_weights/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
ђ
binput_layer/sparse_input_layer/input_layer/C6_embedding/C6_embedding_weights/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
Ѓ
Zinput_layer/sparse_input_layer/input_layer/C6_embedding/C6_embedding_weights/strided_sliceStridedSliceRinput_layer/sparse_input_layer/input_layer/C6_embedding/C6_embedding_weights/Shape`input_layer/sparse_input_layer/input_layer/C6_embedding/C6_embedding_weights/strided_slice/stackbinput_layer/sparse_input_layer/input_layer/C6_embedding/C6_embedding_weights/strided_slice/stack_1binput_layer/sparse_input_layer/input_layer/C6_embedding/C6_embedding_weights/strided_slice/stack_2*
Index0*
shrink_axis_mask*
T0*
_output_shapes
: 
Ц
Tinput_layer/sparse_input_layer/input_layer/C6_embedding/C6_embedding_weights/stack/0Const*
dtype0*
value	B :*
_output_shapes
: 
Ї
Rinput_layer/sparse_input_layer/input_layer/C6_embedding/C6_embedding_weights/stackPackTinput_layer/sparse_input_layer/input_layer/C6_embedding/C6_embedding_weights/stack/0Zinput_layer/sparse_input_layer/input_layer/C6_embedding/C6_embedding_weights/strided_slice*
T0*
N*
_output_shapes
:
ј
Qinput_layer/sparse_input_layer/input_layer/C6_embedding/C6_embedding_weights/TileTileVinput_layer/sparse_input_layer/input_layer/C6_embedding/C6_embedding_weights/Reshape_1Rinput_layer/sparse_input_layer/input_layer/C6_embedding/C6_embedding_weights/stack*
T0
*0
_output_shapes
:€€€€€€€€€€€€€€€€€€
ь
Winput_layer/sparse_input_layer/input_layer/C6_embedding/C6_embedding_weights/zeros_like	ZerosLikedinput_layer/sparse_input_layer/input_layer/C6_embedding/C6_embedding_weights/embedding_lookup_sparse*'
_output_shapes
:€€€€€€€€€*
T0
Ъ
Linput_layer/sparse_input_layer/input_layer/C6_embedding/C6_embedding_weightsSelectQinput_layer/sparse_input_layer/input_layer/C6_embedding/C6_embedding_weights/TileWinput_layer/sparse_input_layer/input_layer/C6_embedding/C6_embedding_weights/zeros_likedinput_layer/sparse_input_layer/input_layer/C6_embedding/C6_embedding_weights/embedding_lookup_sparse*
T0*'
_output_shapes
:€€€€€€€€€
д
Sinput_layer/sparse_input_layer/input_layer/C6_embedding/C6_embedding_weights/Cast_1CastSinput_layer/sparse_input_layer/input_layer/C6_embedding/to_sparse_input/dense_shape*

DstT0*
_output_shapes
:*

SrcT0	
§
Zinput_layer/sparse_input_layer/input_layer/C6_embedding/C6_embedding_weights/Slice_1/beginConst*
_output_shapes
:*
valueB: *
dtype0
£
Yinput_layer/sparse_input_layer/input_layer/C6_embedding/C6_embedding_weights/Slice_1/sizeConst*
dtype0*
valueB:*
_output_shapes
:
Ы
Tinput_layer/sparse_input_layer/input_layer/C6_embedding/C6_embedding_weights/Slice_1SliceSinput_layer/sparse_input_layer/input_layer/C6_embedding/C6_embedding_weights/Cast_1Zinput_layer/sparse_input_layer/input_layer/C6_embedding/C6_embedding_weights/Slice_1/beginYinput_layer/sparse_input_layer/input_layer/C6_embedding/C6_embedding_weights/Slice_1/size*
Index0*
T0*
_output_shapes
:
–
Tinput_layer/sparse_input_layer/input_layer/C6_embedding/C6_embedding_weights/Shape_1ShapeLinput_layer/sparse_input_layer/input_layer/C6_embedding/C6_embedding_weights*
_output_shapes
:*
T0
§
Zinput_layer/sparse_input_layer/input_layer/C6_embedding/C6_embedding_weights/Slice_2/beginConst*
valueB:*
dtype0*
_output_shapes
:
ђ
Yinput_layer/sparse_input_layer/input_layer/C6_embedding/C6_embedding_weights/Slice_2/sizeConst*
dtype0*
_output_shapes
:*
valueB:
€€€€€€€€€
Ь
Tinput_layer/sparse_input_layer/input_layer/C6_embedding/C6_embedding_weights/Slice_2SliceTinput_layer/sparse_input_layer/input_layer/C6_embedding/C6_embedding_weights/Shape_1Zinput_layer/sparse_input_layer/input_layer/C6_embedding/C6_embedding_weights/Slice_2/beginYinput_layer/sparse_input_layer/input_layer/C6_embedding/C6_embedding_weights/Slice_2/size*
_output_shapes
:*
T0*
Index0
Ъ
Xinput_layer/sparse_input_layer/input_layer/C6_embedding/C6_embedding_weights/concat/axisConst*
dtype0*
value	B : *
_output_shapes
: 
У
Sinput_layer/sparse_input_layer/input_layer/C6_embedding/C6_embedding_weights/concatConcatV2Tinput_layer/sparse_input_layer/input_layer/C6_embedding/C6_embedding_weights/Slice_1Tinput_layer/sparse_input_layer/input_layer/C6_embedding/C6_embedding_weights/Slice_2Xinput_layer/sparse_input_layer/input_layer/C6_embedding/C6_embedding_weights/concat/axis*
N*
_output_shapes
:*
T0
ґ
Vinput_layer/sparse_input_layer/input_layer/C6_embedding/C6_embedding_weights/Reshape_2ReshapeLinput_layer/sparse_input_layer/input_layer/C6_embedding/C6_embedding_weightsSinput_layer/sparse_input_layer/input_layer/C6_embedding/C6_embedding_weights/concat*
T0*'
_output_shapes
:€€€€€€€€€
√
=input_layer/sparse_input_layer/input_layer/C6_embedding/ShapeShapeVinput_layer/sparse_input_layer/input_layer/C6_embedding/C6_embedding_weights/Reshape_2*
_output_shapes
:*
T0
Х
Kinput_layer/sparse_input_layer/input_layer/C6_embedding/strided_slice/stackConst*
valueB: *
_output_shapes
:*
dtype0
Ч
Minput_layer/sparse_input_layer/input_layer/C6_embedding/strided_slice/stack_1Const*
dtype0*
valueB:*
_output_shapes
:
Ч
Minput_layer/sparse_input_layer/input_layer/C6_embedding/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
≈
Einput_layer/sparse_input_layer/input_layer/C6_embedding/strided_sliceStridedSlice=input_layer/sparse_input_layer/input_layer/C6_embedding/ShapeKinput_layer/sparse_input_layer/input_layer/C6_embedding/strided_slice/stackMinput_layer/sparse_input_layer/input_layer/C6_embedding/strided_slice/stack_1Minput_layer/sparse_input_layer/input_layer/C6_embedding/strided_slice/stack_2*
shrink_axis_mask*
_output_shapes
: *
T0*
Index0
Й
Ginput_layer/sparse_input_layer/input_layer/C6_embedding/Reshape/shape/1Const*
value	B :*
dtype0*
_output_shapes
: 
Л
Einput_layer/sparse_input_layer/input_layer/C6_embedding/Reshape/shapePackEinput_layer/sparse_input_layer/input_layer/C6_embedding/strided_sliceGinput_layer/sparse_input_layer/input_layer/C6_embedding/Reshape/shape/1*
T0*
_output_shapes
:*
N
Ы
?input_layer/sparse_input_layer/input_layer/C6_embedding/ReshapeReshapeVinput_layer/sparse_input_layer/input_layer/C6_embedding/C6_embedding_weights/Reshape_2Einput_layer/sparse_input_layer/input_layer/C6_embedding/Reshape/shape*'
_output_shapes
:€€€€€€€€€*
T0
С
Finput_layer/sparse_input_layer/input_layer/C7_embedding/ExpandDims/dimConst*
valueB :
€€€€€€€€€*
dtype0*
_output_shapes
: 
Џ
Binput_layer/sparse_input_layer/input_layer/C7_embedding/ExpandDims
ExpandDimsPlaceholder_19Finput_layer/sparse_input_layer/input_layer/C7_embedding/ExpandDims/dim*
T0*'
_output_shapes
:€€€€€€€€€
Ч
Vinput_layer/sparse_input_layer/input_layer/C7_embedding/to_sparse_input/ignore_value/xConst*
dtype0*
_output_shapes
: *
valueB B 
™
Pinput_layer/sparse_input_layer/input_layer/C7_embedding/to_sparse_input/NotEqualNotEqualBinput_layer/sparse_input_layer/input_layer/C7_embedding/ExpandDimsVinput_layer/sparse_input_layer/input_layer/C7_embedding/to_sparse_input/ignore_value/x*'
_output_shapes
:€€€€€€€€€*
T0
”
Oinput_layer/sparse_input_layer/input_layer/C7_embedding/to_sparse_input/indicesWherePinput_layer/sparse_input_layer/input_layer/C7_embedding/to_sparse_input/NotEqual*'
_output_shapes
:€€€€€€€€€
≥
Ninput_layer/sparse_input_layer/input_layer/C7_embedding/to_sparse_input/valuesGatherNdBinput_layer/sparse_input_layer/input_layer/C7_embedding/ExpandDimsOinput_layer/sparse_input_layer/input_layer/C7_embedding/to_sparse_input/indices*#
_output_shapes
:€€€€€€€€€*
Tindices0	*
Tparams0
’
Sinput_layer/sparse_input_layer/input_layer/C7_embedding/to_sparse_input/dense_shapeShapeBinput_layer/sparse_input_layer/input_layer/C7_embedding/ExpandDims*
out_type0	*
T0*
_output_shapes
:
б
>input_layer/sparse_input_layer/input_layer/C7_embedding/lookupStringToHashBucketFastNinput_layer/sparse_input_layer/input_layer/C7_embedding/to_sparse_input/values*#
_output_shapes
:€€€€€€€€€*
num_bucketsРN
Ы
linput_layer/sparse_input_layer/input_layer/C7_embedding/embedding_weights/Initializer/truncated_normal/shapeConst*
dtype0*
valueB"'     *
_output_shapes
:*\
_classR
PNloc:@input_layer/sparse_input_layer/input_layer/C7_embedding/embedding_weights
О
kinput_layer/sparse_input_layer/input_layer/C7_embedding/embedding_weights/Initializer/truncated_normal/meanConst*\
_classR
PNloc:@input_layer/sparse_input_layer/input_layer/C7_embedding/embedding_weights*
valueB
 *    *
dtype0*
_output_shapes
: 
Р
minput_layer/sparse_input_layer/input_layer/C7_embedding/embedding_weights/Initializer/truncated_normal/stddevConst*
valueB
 *  А>*
dtype0*\
_classR
PNloc:@input_layer/sparse_input_layer/input_layer/C7_embedding/embedding_weights*
_output_shapes
: 
М
vinput_layer/sparse_input_layer/input_layer/C7_embedding/embedding_weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormallinput_layer/sparse_input_layer/input_layer/C7_embedding/embedding_weights/Initializer/truncated_normal/shape*
T0*\
_classR
PNloc:@input_layer/sparse_input_layer/input_layer/C7_embedding/embedding_weights*
_output_shapes
:	РN*
dtype0
а
jinput_layer/sparse_input_layer/input_layer/C7_embedding/embedding_weights/Initializer/truncated_normal/mulMulvinput_layer/sparse_input_layer/input_layer/C7_embedding/embedding_weights/Initializer/truncated_normal/TruncatedNormalminput_layer/sparse_input_layer/input_layer/C7_embedding/embedding_weights/Initializer/truncated_normal/stddev*
T0*
_output_shapes
:	РN*\
_classR
PNloc:@input_layer/sparse_input_layer/input_layer/C7_embedding/embedding_weights
ќ
finput_layer/sparse_input_layer/input_layer/C7_embedding/embedding_weights/Initializer/truncated_normalAddjinput_layer/sparse_input_layer/input_layer/C7_embedding/embedding_weights/Initializer/truncated_normal/mulkinput_layer/sparse_input_layer/input_layer/C7_embedding/embedding_weights/Initializer/truncated_normal/mean*
T0*
_output_shapes
:	РN*\
_classR
PNloc:@input_layer/sparse_input_layer/input_layer/C7_embedding/embedding_weights
щ
Iinput_layer/sparse_input_layer/input_layer/C7_embedding/embedding_weights
VariableV2*
_output_shapes
:	РN*
shape:	РN*
dtype0*\
_classR
PNloc:@input_layer/sparse_input_layer/input_layer/C7_embedding/embedding_weights
Х
Pinput_layer/sparse_input_layer/input_layer/C7_embedding/embedding_weights/AssignAssignIinput_layer/sparse_input_layer/input_layer/C7_embedding/embedding_weightsfinput_layer/sparse_input_layer/input_layer/C7_embedding/embedding_weights/Initializer/truncated_normal*\
_classR
PNloc:@input_layer/sparse_input_layer/input_layer/C7_embedding/embedding_weights*
T0*
_output_shapes
:	РN
≠
Ninput_layer/sparse_input_layer/input_layer/C7_embedding/embedding_weights/readIdentityIinput_layer/sparse_input_layer/input_layer/C7_embedding/embedding_weights*
_output_shapes
:	РN*
T0*\
_classR
PNloc:@input_layer/sparse_input_layer/input_layer/C7_embedding/embedding_weights
Ґ
Xinput_layer/sparse_input_layer/input_layer/C7_embedding/C7_embedding_weights/Slice/beginConst*
_output_shapes
:*
valueB: *
dtype0
°
Winput_layer/sparse_input_layer/input_layer/C7_embedding/C7_embedding_weights/Slice/sizeConst*
valueB:*
_output_shapes
:*
dtype0
Х
Rinput_layer/sparse_input_layer/input_layer/C7_embedding/C7_embedding_weights/SliceSliceSinput_layer/sparse_input_layer/input_layer/C7_embedding/to_sparse_input/dense_shapeXinput_layer/sparse_input_layer/input_layer/C7_embedding/C7_embedding_weights/Slice/beginWinput_layer/sparse_input_layer/input_layer/C7_embedding/C7_embedding_weights/Slice/size*
T0	*
Index0*
_output_shapes
:
Ь
Rinput_layer/sparse_input_layer/input_layer/C7_embedding/C7_embedding_weights/ConstConst*
valueB: *
dtype0*
_output_shapes
:
Ґ
Qinput_layer/sparse_input_layer/input_layer/C7_embedding/C7_embedding_weights/ProdProdRinput_layer/sparse_input_layer/input_layer/C7_embedding/C7_embedding_weights/SliceRinput_layer/sparse_input_layer/input_layer/C7_embedding/C7_embedding_weights/Const*
_output_shapes
: *
T0	
Я
]input_layer/sparse_input_layer/input_layer/C7_embedding/C7_embedding_weights/GatherV2/indicesConst*
dtype0*
_output_shapes
: *
value	B :
Ь
Zinput_layer/sparse_input_layer/input_layer/C7_embedding/C7_embedding_weights/GatherV2/axisConst*
dtype0*
value	B : *
_output_shapes
: 
µ
Uinput_layer/sparse_input_layer/input_layer/C7_embedding/C7_embedding_weights/GatherV2GatherV2Sinput_layer/sparse_input_layer/input_layer/C7_embedding/to_sparse_input/dense_shape]input_layer/sparse_input_layer/input_layer/C7_embedding/C7_embedding_weights/GatherV2/indicesZinput_layer/sparse_input_layer/input_layer/C7_embedding/C7_embedding_weights/GatherV2/axis*
Tparams0	*
_output_shapes
: *
Tindices0*
Taxis0
≥
Sinput_layer/sparse_input_layer/input_layer/C7_embedding/C7_embedding_weights/Cast/xPackQinput_layer/sparse_input_layer/input_layer/C7_embedding/C7_embedding_weights/ProdUinput_layer/sparse_input_layer/input_layer/C7_embedding/C7_embedding_weights/GatherV2*
N*
_output_shapes
:*
T0	
Х
Zinput_layer/sparse_input_layer/input_layer/C7_embedding/C7_embedding_weights/SparseReshapeSparseReshapeOinput_layer/sparse_input_layer/input_layer/C7_embedding/to_sparse_input/indicesSinput_layer/sparse_input_layer/input_layer/C7_embedding/to_sparse_input/dense_shapeSinput_layer/sparse_input_layer/input_layer/C7_embedding/C7_embedding_weights/Cast/x*-
_output_shapes
:€€€€€€€€€:
Ё
cinput_layer/sparse_input_layer/input_layer/C7_embedding/C7_embedding_weights/SparseReshape/IdentityIdentity>input_layer/sparse_input_layer/input_layer/C7_embedding/lookup*#
_output_shapes
:€€€€€€€€€*
T0	
Э
[input_layer/sparse_input_layer/input_layer/C7_embedding/C7_embedding_weights/GreaterEqual/yConst*
_output_shapes
: *
dtype0	*
value	B	 R 
ў
Yinput_layer/sparse_input_layer/input_layer/C7_embedding/C7_embedding_weights/GreaterEqualGreaterEqualcinput_layer/sparse_input_layer/input_layer/C7_embedding/C7_embedding_weights/SparseReshape/Identity[input_layer/sparse_input_layer/input_layer/C7_embedding/C7_embedding_weights/GreaterEqual/y*
T0	*#
_output_shapes
:€€€€€€€€€
я
Rinput_layer/sparse_input_layer/input_layer/C7_embedding/C7_embedding_weights/WhereWhereYinput_layer/sparse_input_layer/input_layer/C7_embedding/C7_embedding_weights/GreaterEqual*'
_output_shapes
:€€€€€€€€€
≠
Zinput_layer/sparse_input_layer/input_layer/C7_embedding/C7_embedding_weights/Reshape/shapeConst*
_output_shapes
:*
valueB:
€€€€€€€€€*
dtype0
љ
Tinput_layer/sparse_input_layer/input_layer/C7_embedding/C7_embedding_weights/ReshapeReshapeRinput_layer/sparse_input_layer/input_layer/C7_embedding/C7_embedding_weights/WhereZinput_layer/sparse_input_layer/input_layer/C7_embedding/C7_embedding_weights/Reshape/shape*
T0	*#
_output_shapes
:€€€€€€€€€
Ю
\input_layer/sparse_input_layer/input_layer/C7_embedding/C7_embedding_weights/GatherV2_1/axisConst*
_output_shapes
: *
value	B : *
dtype0
»
Winput_layer/sparse_input_layer/input_layer/C7_embedding/C7_embedding_weights/GatherV2_1GatherV2Zinput_layer/sparse_input_layer/input_layer/C7_embedding/C7_embedding_weights/SparseReshapeTinput_layer/sparse_input_layer/input_layer/C7_embedding/C7_embedding_weights/Reshape\input_layer/sparse_input_layer/input_layer/C7_embedding/C7_embedding_weights/GatherV2_1/axis*
Taxis0*
Tparams0	*'
_output_shapes
:€€€€€€€€€*
Tindices0	
Ю
\input_layer/sparse_input_layer/input_layer/C7_embedding/C7_embedding_weights/GatherV2_2/axisConst*
_output_shapes
: *
value	B : *
dtype0
Ќ
Winput_layer/sparse_input_layer/input_layer/C7_embedding/C7_embedding_weights/GatherV2_2GatherV2cinput_layer/sparse_input_layer/input_layer/C7_embedding/C7_embedding_weights/SparseReshape/IdentityTinput_layer/sparse_input_layer/input_layer/C7_embedding/C7_embedding_weights/Reshape\input_layer/sparse_input_layer/input_layer/C7_embedding/C7_embedding_weights/GatherV2_2/axis*
Tparams0	*#
_output_shapes
:€€€€€€€€€*
Tindices0	*
Taxis0
д
Uinput_layer/sparse_input_layer/input_layer/C7_embedding/C7_embedding_weights/IdentityIdentity\input_layer/sparse_input_layer/input_layer/C7_embedding/C7_embedding_weights/SparseReshape:1*
T0	*
_output_shapes
:
®
finput_layer/sparse_input_layer/input_layer/C7_embedding/C7_embedding_weights/SparseFillEmptyRows/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R 
џ
tinput_layer/sparse_input_layer/input_layer/C7_embedding/C7_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowsSparseFillEmptyRowsWinput_layer/sparse_input_layer/input_layer/C7_embedding/C7_embedding_weights/GatherV2_1Winput_layer/sparse_input_layer/input_layer/C7_embedding/C7_embedding_weights/GatherV2_2Uinput_layer/sparse_input_layer/input_layer/C7_embedding/C7_embedding_weights/Identityfinput_layer/sparse_input_layer/input_layer/C7_embedding/C7_embedding_weights/SparseFillEmptyRows/Const*T
_output_shapesB
@:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€*
T0	
…
xinput_layer/sparse_input_layer/input_layer/C7_embedding/C7_embedding_weights/embedding_lookup_sparse/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB"        
Ћ
zinput_layer/sparse_input_layer/input_layer/C7_embedding/C7_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1Const*
dtype0*
valueB"       *
_output_shapes
:
Ћ
zinput_layer/sparse_input_layer/input_layer/C7_embedding/C7_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2Const*
_output_shapes
:*
valueB"      *
dtype0
я
rinput_layer/sparse_input_layer/input_layer/C7_embedding/C7_embedding_weights/embedding_lookup_sparse/strided_sliceStridedSlicetinput_layer/sparse_input_layer/input_layer/C7_embedding/C7_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowsxinput_layer/sparse_input_layer/input_layer/C7_embedding/C7_embedding_weights/embedding_lookup_sparse/strided_slice/stackzinput_layer/sparse_input_layer/input_layer/C7_embedding/C7_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1zinput_layer/sparse_input_layer/input_layer/C7_embedding/C7_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2*#
_output_shapes
:€€€€€€€€€*

begin_mask*
shrink_axis_mask*
end_mask*
T0	*
Index0
Ґ
iinput_layer/sparse_input_layer/input_layer/C7_embedding/C7_embedding_weights/embedding_lookup_sparse/CastCastrinput_layer/sparse_input_layer/input_layer/C7_embedding/C7_embedding_weights/embedding_lookup_sparse/strided_slice*

SrcT0	*#
_output_shapes
:€€€€€€€€€*

DstT0
™
kinput_layer/sparse_input_layer/input_layer/C7_embedding/C7_embedding_weights/embedding_lookup_sparse/UniqueUniquevinput_layer/sparse_input_layer/input_layer/C7_embedding/C7_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€*
T0	
Ъ
zinput_layer/sparse_input_layer/input_layer/C7_embedding/C7_embedding_weights/embedding_lookup_sparse/embedding_lookup/axisConst*
_output_shapes
: *\
_classR
PNloc:@input_layer/sparse_input_layer/input_layer/C7_embedding/embedding_weights*
value	B : *
dtype0
н
uinput_layer/sparse_input_layer/input_layer/C7_embedding/C7_embedding_weights/embedding_lookup_sparse/embedding_lookupGatherV2Ninput_layer/sparse_input_layer/input_layer/C7_embedding/embedding_weights/readkinput_layer/sparse_input_layer/input_layer/C7_embedding/C7_embedding_weights/embedding_lookup_sparse/Uniquezinput_layer/sparse_input_layer/input_layer/C7_embedding/C7_embedding_weights/embedding_lookup_sparse/embedding_lookup/axis*
Tparams0*
Tindices0	*'
_output_shapes
:€€€€€€€€€*
Taxis0*\
_classR
PNloc:@input_layer/sparse_input_layer/input_layer/C7_embedding/embedding_weights
≥
~input_layer/sparse_input_layer/input_layer/C7_embedding/C7_embedding_weights/embedding_lookup_sparse/embedding_lookup/IdentityIdentityuinput_layer/sparse_input_layer/input_layer/C7_embedding/C7_embedding_weights/embedding_lookup_sparse/embedding_lookup*
T0*'
_output_shapes
:€€€€€€€€€
Е
dinput_layer/sparse_input_layer/input_layer/C7_embedding/C7_embedding_weights/embedding_lookup_sparseSparseSegmentMean~input_layer/sparse_input_layer/input_layer/C7_embedding/C7_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identityminput_layer/sparse_input_layer/input_layer/C7_embedding/C7_embedding_weights/embedding_lookup_sparse/Unique:1iinput_layer/sparse_input_layer/input_layer/C7_embedding/C7_embedding_weights/embedding_lookup_sparse/Cast*'
_output_shapes
:€€€€€€€€€*
T0
≠
\input_layer/sparse_input_layer/input_layer/C7_embedding/C7_embedding_weights/Reshape_1/shapeConst*
dtype0*
valueB"€€€€   *
_output_shapes
:
й
Vinput_layer/sparse_input_layer/input_layer/C7_embedding/C7_embedding_weights/Reshape_1Reshapevinput_layer/sparse_input_layer/input_layer/C7_embedding/C7_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:2\input_layer/sparse_input_layer/input_layer/C7_embedding/C7_embedding_weights/Reshape_1/shape*
T0
*'
_output_shapes
:€€€€€€€€€
ж
Rinput_layer/sparse_input_layer/input_layer/C7_embedding/C7_embedding_weights/ShapeShapedinput_layer/sparse_input_layer/input_layer/C7_embedding/C7_embedding_weights/embedding_lookup_sparse*
_output_shapes
:*
T0
™
`input_layer/sparse_input_layer/input_layer/C7_embedding/C7_embedding_weights/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
ђ
binput_layer/sparse_input_layer/input_layer/C7_embedding/C7_embedding_weights/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
ђ
binput_layer/sparse_input_layer/input_layer/C7_embedding/C7_embedding_weights/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
Ѓ
Zinput_layer/sparse_input_layer/input_layer/C7_embedding/C7_embedding_weights/strided_sliceStridedSliceRinput_layer/sparse_input_layer/input_layer/C7_embedding/C7_embedding_weights/Shape`input_layer/sparse_input_layer/input_layer/C7_embedding/C7_embedding_weights/strided_slice/stackbinput_layer/sparse_input_layer/input_layer/C7_embedding/C7_embedding_weights/strided_slice/stack_1binput_layer/sparse_input_layer/input_layer/C7_embedding/C7_embedding_weights/strided_slice/stack_2*
Index0*
T0*
shrink_axis_mask*
_output_shapes
: 
Ц
Tinput_layer/sparse_input_layer/input_layer/C7_embedding/C7_embedding_weights/stack/0Const*
dtype0*
value	B :*
_output_shapes
: 
Ї
Rinput_layer/sparse_input_layer/input_layer/C7_embedding/C7_embedding_weights/stackPackTinput_layer/sparse_input_layer/input_layer/C7_embedding/C7_embedding_weights/stack/0Zinput_layer/sparse_input_layer/input_layer/C7_embedding/C7_embedding_weights/strided_slice*
T0*
N*
_output_shapes
:
ј
Qinput_layer/sparse_input_layer/input_layer/C7_embedding/C7_embedding_weights/TileTileVinput_layer/sparse_input_layer/input_layer/C7_embedding/C7_embedding_weights/Reshape_1Rinput_layer/sparse_input_layer/input_layer/C7_embedding/C7_embedding_weights/stack*
T0
*0
_output_shapes
:€€€€€€€€€€€€€€€€€€
ь
Winput_layer/sparse_input_layer/input_layer/C7_embedding/C7_embedding_weights/zeros_like	ZerosLikedinput_layer/sparse_input_layer/input_layer/C7_embedding/C7_embedding_weights/embedding_lookup_sparse*'
_output_shapes
:€€€€€€€€€*
T0
Ъ
Linput_layer/sparse_input_layer/input_layer/C7_embedding/C7_embedding_weightsSelectQinput_layer/sparse_input_layer/input_layer/C7_embedding/C7_embedding_weights/TileWinput_layer/sparse_input_layer/input_layer/C7_embedding/C7_embedding_weights/zeros_likedinput_layer/sparse_input_layer/input_layer/C7_embedding/C7_embedding_weights/embedding_lookup_sparse*
T0*'
_output_shapes
:€€€€€€€€€
д
Sinput_layer/sparse_input_layer/input_layer/C7_embedding/C7_embedding_weights/Cast_1CastSinput_layer/sparse_input_layer/input_layer/C7_embedding/to_sparse_input/dense_shape*
_output_shapes
:*

SrcT0	*

DstT0
§
Zinput_layer/sparse_input_layer/input_layer/C7_embedding/C7_embedding_weights/Slice_1/beginConst*
_output_shapes
:*
valueB: *
dtype0
£
Yinput_layer/sparse_input_layer/input_layer/C7_embedding/C7_embedding_weights/Slice_1/sizeConst*
valueB:*
dtype0*
_output_shapes
:
Ы
Tinput_layer/sparse_input_layer/input_layer/C7_embedding/C7_embedding_weights/Slice_1SliceSinput_layer/sparse_input_layer/input_layer/C7_embedding/C7_embedding_weights/Cast_1Zinput_layer/sparse_input_layer/input_layer/C7_embedding/C7_embedding_weights/Slice_1/beginYinput_layer/sparse_input_layer/input_layer/C7_embedding/C7_embedding_weights/Slice_1/size*
_output_shapes
:*
Index0*
T0
–
Tinput_layer/sparse_input_layer/input_layer/C7_embedding/C7_embedding_weights/Shape_1ShapeLinput_layer/sparse_input_layer/input_layer/C7_embedding/C7_embedding_weights*
_output_shapes
:*
T0
§
Zinput_layer/sparse_input_layer/input_layer/C7_embedding/C7_embedding_weights/Slice_2/beginConst*
_output_shapes
:*
valueB:*
dtype0
ђ
Yinput_layer/sparse_input_layer/input_layer/C7_embedding/C7_embedding_weights/Slice_2/sizeConst*
dtype0*
_output_shapes
:*
valueB:
€€€€€€€€€
Ь
Tinput_layer/sparse_input_layer/input_layer/C7_embedding/C7_embedding_weights/Slice_2SliceTinput_layer/sparse_input_layer/input_layer/C7_embedding/C7_embedding_weights/Shape_1Zinput_layer/sparse_input_layer/input_layer/C7_embedding/C7_embedding_weights/Slice_2/beginYinput_layer/sparse_input_layer/input_layer/C7_embedding/C7_embedding_weights/Slice_2/size*
Index0*
_output_shapes
:*
T0
Ъ
Xinput_layer/sparse_input_layer/input_layer/C7_embedding/C7_embedding_weights/concat/axisConst*
value	B : *
_output_shapes
: *
dtype0
У
Sinput_layer/sparse_input_layer/input_layer/C7_embedding/C7_embedding_weights/concatConcatV2Tinput_layer/sparse_input_layer/input_layer/C7_embedding/C7_embedding_weights/Slice_1Tinput_layer/sparse_input_layer/input_layer/C7_embedding/C7_embedding_weights/Slice_2Xinput_layer/sparse_input_layer/input_layer/C7_embedding/C7_embedding_weights/concat/axis*
N*
_output_shapes
:*
T0
ґ
Vinput_layer/sparse_input_layer/input_layer/C7_embedding/C7_embedding_weights/Reshape_2ReshapeLinput_layer/sparse_input_layer/input_layer/C7_embedding/C7_embedding_weightsSinput_layer/sparse_input_layer/input_layer/C7_embedding/C7_embedding_weights/concat*
T0*'
_output_shapes
:€€€€€€€€€
√
=input_layer/sparse_input_layer/input_layer/C7_embedding/ShapeShapeVinput_layer/sparse_input_layer/input_layer/C7_embedding/C7_embedding_weights/Reshape_2*
_output_shapes
:*
T0
Х
Kinput_layer/sparse_input_layer/input_layer/C7_embedding/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
Ч
Minput_layer/sparse_input_layer/input_layer/C7_embedding/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Ч
Minput_layer/sparse_input_layer/input_layer/C7_embedding/strided_slice/stack_2Const*
valueB:*
_output_shapes
:*
dtype0
≈
Einput_layer/sparse_input_layer/input_layer/C7_embedding/strided_sliceStridedSlice=input_layer/sparse_input_layer/input_layer/C7_embedding/ShapeKinput_layer/sparse_input_layer/input_layer/C7_embedding/strided_slice/stackMinput_layer/sparse_input_layer/input_layer/C7_embedding/strided_slice/stack_1Minput_layer/sparse_input_layer/input_layer/C7_embedding/strided_slice/stack_2*
T0*
_output_shapes
: *
shrink_axis_mask*
Index0
Й
Ginput_layer/sparse_input_layer/input_layer/C7_embedding/Reshape/shape/1Const*
value	B :*
dtype0*
_output_shapes
: 
Л
Einput_layer/sparse_input_layer/input_layer/C7_embedding/Reshape/shapePackEinput_layer/sparse_input_layer/input_layer/C7_embedding/strided_sliceGinput_layer/sparse_input_layer/input_layer/C7_embedding/Reshape/shape/1*
_output_shapes
:*
N*
T0
Ы
?input_layer/sparse_input_layer/input_layer/C7_embedding/ReshapeReshapeVinput_layer/sparse_input_layer/input_layer/C7_embedding/C7_embedding_weights/Reshape_2Einput_layer/sparse_input_layer/input_layer/C7_embedding/Reshape/shape*'
_output_shapes
:€€€€€€€€€*
T0
С
Finput_layer/sparse_input_layer/input_layer/C8_embedding/ExpandDims/dimConst*
dtype0*
valueB :
€€€€€€€€€*
_output_shapes
: 
Џ
Binput_layer/sparse_input_layer/input_layer/C8_embedding/ExpandDims
ExpandDimsPlaceholder_20Finput_layer/sparse_input_layer/input_layer/C8_embedding/ExpandDims/dim*'
_output_shapes
:€€€€€€€€€*
T0
Ч
Vinput_layer/sparse_input_layer/input_layer/C8_embedding/to_sparse_input/ignore_value/xConst*
dtype0*
valueB B *
_output_shapes
: 
™
Pinput_layer/sparse_input_layer/input_layer/C8_embedding/to_sparse_input/NotEqualNotEqualBinput_layer/sparse_input_layer/input_layer/C8_embedding/ExpandDimsVinput_layer/sparse_input_layer/input_layer/C8_embedding/to_sparse_input/ignore_value/x*
T0*'
_output_shapes
:€€€€€€€€€
”
Oinput_layer/sparse_input_layer/input_layer/C8_embedding/to_sparse_input/indicesWherePinput_layer/sparse_input_layer/input_layer/C8_embedding/to_sparse_input/NotEqual*'
_output_shapes
:€€€€€€€€€
≥
Ninput_layer/sparse_input_layer/input_layer/C8_embedding/to_sparse_input/valuesGatherNdBinput_layer/sparse_input_layer/input_layer/C8_embedding/ExpandDimsOinput_layer/sparse_input_layer/input_layer/C8_embedding/to_sparse_input/indices*
Tparams0*
Tindices0	*#
_output_shapes
:€€€€€€€€€
’
Sinput_layer/sparse_input_layer/input_layer/C8_embedding/to_sparse_input/dense_shapeShapeBinput_layer/sparse_input_layer/input_layer/C8_embedding/ExpandDims*
out_type0	*
T0*
_output_shapes
:
б
>input_layer/sparse_input_layer/input_layer/C8_embedding/lookupStringToHashBucketFastNinput_layer/sparse_input_layer/input_layer/C8_embedding/to_sparse_input/values*#
_output_shapes
:€€€€€€€€€*
num_bucketsРN
Ы
linput_layer/sparse_input_layer/input_layer/C8_embedding/embedding_weights/Initializer/truncated_normal/shapeConst*
dtype0*\
_classR
PNloc:@input_layer/sparse_input_layer/input_layer/C8_embedding/embedding_weights*
_output_shapes
:*
valueB"'     
О
kinput_layer/sparse_input_layer/input_layer/C8_embedding/embedding_weights/Initializer/truncated_normal/meanConst*\
_classR
PNloc:@input_layer/sparse_input_layer/input_layer/C8_embedding/embedding_weights*
valueB
 *    *
dtype0*
_output_shapes
: 
Р
minput_layer/sparse_input_layer/input_layer/C8_embedding/embedding_weights/Initializer/truncated_normal/stddevConst*
valueB
 *  А>*\
_classR
PNloc:@input_layer/sparse_input_layer/input_layer/C8_embedding/embedding_weights*
dtype0*
_output_shapes
: 
М
vinput_layer/sparse_input_layer/input_layer/C8_embedding/embedding_weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormallinput_layer/sparse_input_layer/input_layer/C8_embedding/embedding_weights/Initializer/truncated_normal/shape*\
_classR
PNloc:@input_layer/sparse_input_layer/input_layer/C8_embedding/embedding_weights*
T0*
dtype0*
_output_shapes
:	РN
а
jinput_layer/sparse_input_layer/input_layer/C8_embedding/embedding_weights/Initializer/truncated_normal/mulMulvinput_layer/sparse_input_layer/input_layer/C8_embedding/embedding_weights/Initializer/truncated_normal/TruncatedNormalminput_layer/sparse_input_layer/input_layer/C8_embedding/embedding_weights/Initializer/truncated_normal/stddev*
T0*
_output_shapes
:	РN*\
_classR
PNloc:@input_layer/sparse_input_layer/input_layer/C8_embedding/embedding_weights
ќ
finput_layer/sparse_input_layer/input_layer/C8_embedding/embedding_weights/Initializer/truncated_normalAddjinput_layer/sparse_input_layer/input_layer/C8_embedding/embedding_weights/Initializer/truncated_normal/mulkinput_layer/sparse_input_layer/input_layer/C8_embedding/embedding_weights/Initializer/truncated_normal/mean*\
_classR
PNloc:@input_layer/sparse_input_layer/input_layer/C8_embedding/embedding_weights*
_output_shapes
:	РN*
T0
щ
Iinput_layer/sparse_input_layer/input_layer/C8_embedding/embedding_weights
VariableV2*\
_classR
PNloc:@input_layer/sparse_input_layer/input_layer/C8_embedding/embedding_weights*
_output_shapes
:	РN*
shape:	РN*
dtype0
Х
Pinput_layer/sparse_input_layer/input_layer/C8_embedding/embedding_weights/AssignAssignIinput_layer/sparse_input_layer/input_layer/C8_embedding/embedding_weightsfinput_layer/sparse_input_layer/input_layer/C8_embedding/embedding_weights/Initializer/truncated_normal*\
_classR
PNloc:@input_layer/sparse_input_layer/input_layer/C8_embedding/embedding_weights*
T0*
_output_shapes
:	РN
≠
Ninput_layer/sparse_input_layer/input_layer/C8_embedding/embedding_weights/readIdentityIinput_layer/sparse_input_layer/input_layer/C8_embedding/embedding_weights*
T0*
_output_shapes
:	РN*\
_classR
PNloc:@input_layer/sparse_input_layer/input_layer/C8_embedding/embedding_weights
Ґ
Xinput_layer/sparse_input_layer/input_layer/C8_embedding/C8_embedding_weights/Slice/beginConst*
_output_shapes
:*
dtype0*
valueB: 
°
Winput_layer/sparse_input_layer/input_layer/C8_embedding/C8_embedding_weights/Slice/sizeConst*
valueB:*
_output_shapes
:*
dtype0
Х
Rinput_layer/sparse_input_layer/input_layer/C8_embedding/C8_embedding_weights/SliceSliceSinput_layer/sparse_input_layer/input_layer/C8_embedding/to_sparse_input/dense_shapeXinput_layer/sparse_input_layer/input_layer/C8_embedding/C8_embedding_weights/Slice/beginWinput_layer/sparse_input_layer/input_layer/C8_embedding/C8_embedding_weights/Slice/size*
_output_shapes
:*
Index0*
T0	
Ь
Rinput_layer/sparse_input_layer/input_layer/C8_embedding/C8_embedding_weights/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
Ґ
Qinput_layer/sparse_input_layer/input_layer/C8_embedding/C8_embedding_weights/ProdProdRinput_layer/sparse_input_layer/input_layer/C8_embedding/C8_embedding_weights/SliceRinput_layer/sparse_input_layer/input_layer/C8_embedding/C8_embedding_weights/Const*
_output_shapes
: *
T0	
Я
]input_layer/sparse_input_layer/input_layer/C8_embedding/C8_embedding_weights/GatherV2/indicesConst*
_output_shapes
: *
value	B :*
dtype0
Ь
Zinput_layer/sparse_input_layer/input_layer/C8_embedding/C8_embedding_weights/GatherV2/axisConst*
dtype0*
value	B : *
_output_shapes
: 
µ
Uinput_layer/sparse_input_layer/input_layer/C8_embedding/C8_embedding_weights/GatherV2GatherV2Sinput_layer/sparse_input_layer/input_layer/C8_embedding/to_sparse_input/dense_shape]input_layer/sparse_input_layer/input_layer/C8_embedding/C8_embedding_weights/GatherV2/indicesZinput_layer/sparse_input_layer/input_layer/C8_embedding/C8_embedding_weights/GatherV2/axis*
Taxis0*
Tindices0*
Tparams0	*
_output_shapes
: 
≥
Sinput_layer/sparse_input_layer/input_layer/C8_embedding/C8_embedding_weights/Cast/xPackQinput_layer/sparse_input_layer/input_layer/C8_embedding/C8_embedding_weights/ProdUinput_layer/sparse_input_layer/input_layer/C8_embedding/C8_embedding_weights/GatherV2*
N*
_output_shapes
:*
T0	
Х
Zinput_layer/sparse_input_layer/input_layer/C8_embedding/C8_embedding_weights/SparseReshapeSparseReshapeOinput_layer/sparse_input_layer/input_layer/C8_embedding/to_sparse_input/indicesSinput_layer/sparse_input_layer/input_layer/C8_embedding/to_sparse_input/dense_shapeSinput_layer/sparse_input_layer/input_layer/C8_embedding/C8_embedding_weights/Cast/x*-
_output_shapes
:€€€€€€€€€:
Ё
cinput_layer/sparse_input_layer/input_layer/C8_embedding/C8_embedding_weights/SparseReshape/IdentityIdentity>input_layer/sparse_input_layer/input_layer/C8_embedding/lookup*#
_output_shapes
:€€€€€€€€€*
T0	
Э
[input_layer/sparse_input_layer/input_layer/C8_embedding/C8_embedding_weights/GreaterEqual/yConst*
dtype0	*
value	B	 R *
_output_shapes
: 
ў
Yinput_layer/sparse_input_layer/input_layer/C8_embedding/C8_embedding_weights/GreaterEqualGreaterEqualcinput_layer/sparse_input_layer/input_layer/C8_embedding/C8_embedding_weights/SparseReshape/Identity[input_layer/sparse_input_layer/input_layer/C8_embedding/C8_embedding_weights/GreaterEqual/y*
T0	*#
_output_shapes
:€€€€€€€€€
я
Rinput_layer/sparse_input_layer/input_layer/C8_embedding/C8_embedding_weights/WhereWhereYinput_layer/sparse_input_layer/input_layer/C8_embedding/C8_embedding_weights/GreaterEqual*'
_output_shapes
:€€€€€€€€€
≠
Zinput_layer/sparse_input_layer/input_layer/C8_embedding/C8_embedding_weights/Reshape/shapeConst*
dtype0*
valueB:
€€€€€€€€€*
_output_shapes
:
љ
Tinput_layer/sparse_input_layer/input_layer/C8_embedding/C8_embedding_weights/ReshapeReshapeRinput_layer/sparse_input_layer/input_layer/C8_embedding/C8_embedding_weights/WhereZinput_layer/sparse_input_layer/input_layer/C8_embedding/C8_embedding_weights/Reshape/shape*
T0	*#
_output_shapes
:€€€€€€€€€
Ю
\input_layer/sparse_input_layer/input_layer/C8_embedding/C8_embedding_weights/GatherV2_1/axisConst*
dtype0*
_output_shapes
: *
value	B : 
»
Winput_layer/sparse_input_layer/input_layer/C8_embedding/C8_embedding_weights/GatherV2_1GatherV2Zinput_layer/sparse_input_layer/input_layer/C8_embedding/C8_embedding_weights/SparseReshapeTinput_layer/sparse_input_layer/input_layer/C8_embedding/C8_embedding_weights/Reshape\input_layer/sparse_input_layer/input_layer/C8_embedding/C8_embedding_weights/GatherV2_1/axis*'
_output_shapes
:€€€€€€€€€*
Tparams0	*
Taxis0*
Tindices0	
Ю
\input_layer/sparse_input_layer/input_layer/C8_embedding/C8_embedding_weights/GatherV2_2/axisConst*
value	B : *
_output_shapes
: *
dtype0
Ќ
Winput_layer/sparse_input_layer/input_layer/C8_embedding/C8_embedding_weights/GatherV2_2GatherV2cinput_layer/sparse_input_layer/input_layer/C8_embedding/C8_embedding_weights/SparseReshape/IdentityTinput_layer/sparse_input_layer/input_layer/C8_embedding/C8_embedding_weights/Reshape\input_layer/sparse_input_layer/input_layer/C8_embedding/C8_embedding_weights/GatherV2_2/axis*
Tparams0	*
Taxis0*
Tindices0	*#
_output_shapes
:€€€€€€€€€
д
Uinput_layer/sparse_input_layer/input_layer/C8_embedding/C8_embedding_weights/IdentityIdentity\input_layer/sparse_input_layer/input_layer/C8_embedding/C8_embedding_weights/SparseReshape:1*
T0	*
_output_shapes
:
®
finput_layer/sparse_input_layer/input_layer/C8_embedding/C8_embedding_weights/SparseFillEmptyRows/ConstConst*
dtype0	*
value	B	 R *
_output_shapes
: 
џ
tinput_layer/sparse_input_layer/input_layer/C8_embedding/C8_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowsSparseFillEmptyRowsWinput_layer/sparse_input_layer/input_layer/C8_embedding/C8_embedding_weights/GatherV2_1Winput_layer/sparse_input_layer/input_layer/C8_embedding/C8_embedding_weights/GatherV2_2Uinput_layer/sparse_input_layer/input_layer/C8_embedding/C8_embedding_weights/Identityfinput_layer/sparse_input_layer/input_layer/C8_embedding/C8_embedding_weights/SparseFillEmptyRows/Const*
T0	*T
_output_shapesB
@:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€
…
xinput_layer/sparse_input_layer/input_layer/C8_embedding/C8_embedding_weights/embedding_lookup_sparse/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB"        
Ћ
zinput_layer/sparse_input_layer/input_layer/C8_embedding/C8_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1Const*
_output_shapes
:*
valueB"       *
dtype0
Ћ
zinput_layer/sparse_input_layer/input_layer/C8_embedding/C8_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:
я
rinput_layer/sparse_input_layer/input_layer/C8_embedding/C8_embedding_weights/embedding_lookup_sparse/strided_sliceStridedSlicetinput_layer/sparse_input_layer/input_layer/C8_embedding/C8_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowsxinput_layer/sparse_input_layer/input_layer/C8_embedding/C8_embedding_weights/embedding_lookup_sparse/strided_slice/stackzinput_layer/sparse_input_layer/input_layer/C8_embedding/C8_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1zinput_layer/sparse_input_layer/input_layer/C8_embedding/C8_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2*#
_output_shapes
:€€€€€€€€€*
end_mask*
shrink_axis_mask*

begin_mask*
T0	*
Index0
Ґ
iinput_layer/sparse_input_layer/input_layer/C8_embedding/C8_embedding_weights/embedding_lookup_sparse/CastCastrinput_layer/sparse_input_layer/input_layer/C8_embedding/C8_embedding_weights/embedding_lookup_sparse/strided_slice*#
_output_shapes
:€€€€€€€€€*

DstT0*

SrcT0	
™
kinput_layer/sparse_input_layer/input_layer/C8_embedding/C8_embedding_weights/embedding_lookup_sparse/UniqueUniquevinput_layer/sparse_input_layer/input_layer/C8_embedding/C8_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:1*
T0	*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
Ъ
zinput_layer/sparse_input_layer/input_layer/C8_embedding/C8_embedding_weights/embedding_lookup_sparse/embedding_lookup/axisConst*
dtype0*
value	B : *
_output_shapes
: *\
_classR
PNloc:@input_layer/sparse_input_layer/input_layer/C8_embedding/embedding_weights
н
uinput_layer/sparse_input_layer/input_layer/C8_embedding/C8_embedding_weights/embedding_lookup_sparse/embedding_lookupGatherV2Ninput_layer/sparse_input_layer/input_layer/C8_embedding/embedding_weights/readkinput_layer/sparse_input_layer/input_layer/C8_embedding/C8_embedding_weights/embedding_lookup_sparse/Uniquezinput_layer/sparse_input_layer/input_layer/C8_embedding/C8_embedding_weights/embedding_lookup_sparse/embedding_lookup/axis*\
_classR
PNloc:@input_layer/sparse_input_layer/input_layer/C8_embedding/embedding_weights*
Taxis0*
Tindices0	*
Tparams0*'
_output_shapes
:€€€€€€€€€
≥
~input_layer/sparse_input_layer/input_layer/C8_embedding/C8_embedding_weights/embedding_lookup_sparse/embedding_lookup/IdentityIdentityuinput_layer/sparse_input_layer/input_layer/C8_embedding/C8_embedding_weights/embedding_lookup_sparse/embedding_lookup*
T0*'
_output_shapes
:€€€€€€€€€
Е
dinput_layer/sparse_input_layer/input_layer/C8_embedding/C8_embedding_weights/embedding_lookup_sparseSparseSegmentMean~input_layer/sparse_input_layer/input_layer/C8_embedding/C8_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identityminput_layer/sparse_input_layer/input_layer/C8_embedding/C8_embedding_weights/embedding_lookup_sparse/Unique:1iinput_layer/sparse_input_layer/input_layer/C8_embedding/C8_embedding_weights/embedding_lookup_sparse/Cast*
T0*'
_output_shapes
:€€€€€€€€€
≠
\input_layer/sparse_input_layer/input_layer/C8_embedding/C8_embedding_weights/Reshape_1/shapeConst*
dtype0*
_output_shapes
:*
valueB"€€€€   
й
Vinput_layer/sparse_input_layer/input_layer/C8_embedding/C8_embedding_weights/Reshape_1Reshapevinput_layer/sparse_input_layer/input_layer/C8_embedding/C8_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:2\input_layer/sparse_input_layer/input_layer/C8_embedding/C8_embedding_weights/Reshape_1/shape*
T0
*'
_output_shapes
:€€€€€€€€€
ж
Rinput_layer/sparse_input_layer/input_layer/C8_embedding/C8_embedding_weights/ShapeShapedinput_layer/sparse_input_layer/input_layer/C8_embedding/C8_embedding_weights/embedding_lookup_sparse*
_output_shapes
:*
T0
™
`input_layer/sparse_input_layer/input_layer/C8_embedding/C8_embedding_weights/strided_slice/stackConst*
dtype0*
valueB:*
_output_shapes
:
ђ
binput_layer/sparse_input_layer/input_layer/C8_embedding/C8_embedding_weights/strided_slice/stack_1Const*
dtype0*
valueB:*
_output_shapes
:
ђ
binput_layer/sparse_input_layer/input_layer/C8_embedding/C8_embedding_weights/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Ѓ
Zinput_layer/sparse_input_layer/input_layer/C8_embedding/C8_embedding_weights/strided_sliceStridedSliceRinput_layer/sparse_input_layer/input_layer/C8_embedding/C8_embedding_weights/Shape`input_layer/sparse_input_layer/input_layer/C8_embedding/C8_embedding_weights/strided_slice/stackbinput_layer/sparse_input_layer/input_layer/C8_embedding/C8_embedding_weights/strided_slice/stack_1binput_layer/sparse_input_layer/input_layer/C8_embedding/C8_embedding_weights/strided_slice/stack_2*
_output_shapes
: *
Index0*
shrink_axis_mask*
T0
Ц
Tinput_layer/sparse_input_layer/input_layer/C8_embedding/C8_embedding_weights/stack/0Const*
dtype0*
_output_shapes
: *
value	B :
Ї
Rinput_layer/sparse_input_layer/input_layer/C8_embedding/C8_embedding_weights/stackPackTinput_layer/sparse_input_layer/input_layer/C8_embedding/C8_embedding_weights/stack/0Zinput_layer/sparse_input_layer/input_layer/C8_embedding/C8_embedding_weights/strided_slice*
_output_shapes
:*
T0*
N
ј
Qinput_layer/sparse_input_layer/input_layer/C8_embedding/C8_embedding_weights/TileTileVinput_layer/sparse_input_layer/input_layer/C8_embedding/C8_embedding_weights/Reshape_1Rinput_layer/sparse_input_layer/input_layer/C8_embedding/C8_embedding_weights/stack*
T0
*0
_output_shapes
:€€€€€€€€€€€€€€€€€€
ь
Winput_layer/sparse_input_layer/input_layer/C8_embedding/C8_embedding_weights/zeros_like	ZerosLikedinput_layer/sparse_input_layer/input_layer/C8_embedding/C8_embedding_weights/embedding_lookup_sparse*
T0*'
_output_shapes
:€€€€€€€€€
Ъ
Linput_layer/sparse_input_layer/input_layer/C8_embedding/C8_embedding_weightsSelectQinput_layer/sparse_input_layer/input_layer/C8_embedding/C8_embedding_weights/TileWinput_layer/sparse_input_layer/input_layer/C8_embedding/C8_embedding_weights/zeros_likedinput_layer/sparse_input_layer/input_layer/C8_embedding/C8_embedding_weights/embedding_lookup_sparse*'
_output_shapes
:€€€€€€€€€*
T0
д
Sinput_layer/sparse_input_layer/input_layer/C8_embedding/C8_embedding_weights/Cast_1CastSinput_layer/sparse_input_layer/input_layer/C8_embedding/to_sparse_input/dense_shape*

DstT0*

SrcT0	*
_output_shapes
:
§
Zinput_layer/sparse_input_layer/input_layer/C8_embedding/C8_embedding_weights/Slice_1/beginConst*
_output_shapes
:*
dtype0*
valueB: 
£
Yinput_layer/sparse_input_layer/input_layer/C8_embedding/C8_embedding_weights/Slice_1/sizeConst*
_output_shapes
:*
valueB:*
dtype0
Ы
Tinput_layer/sparse_input_layer/input_layer/C8_embedding/C8_embedding_weights/Slice_1SliceSinput_layer/sparse_input_layer/input_layer/C8_embedding/C8_embedding_weights/Cast_1Zinput_layer/sparse_input_layer/input_layer/C8_embedding/C8_embedding_weights/Slice_1/beginYinput_layer/sparse_input_layer/input_layer/C8_embedding/C8_embedding_weights/Slice_1/size*
T0*
Index0*
_output_shapes
:
–
Tinput_layer/sparse_input_layer/input_layer/C8_embedding/C8_embedding_weights/Shape_1ShapeLinput_layer/sparse_input_layer/input_layer/C8_embedding/C8_embedding_weights*
_output_shapes
:*
T0
§
Zinput_layer/sparse_input_layer/input_layer/C8_embedding/C8_embedding_weights/Slice_2/beginConst*
valueB:*
_output_shapes
:*
dtype0
ђ
Yinput_layer/sparse_input_layer/input_layer/C8_embedding/C8_embedding_weights/Slice_2/sizeConst*
dtype0*
_output_shapes
:*
valueB:
€€€€€€€€€
Ь
Tinput_layer/sparse_input_layer/input_layer/C8_embedding/C8_embedding_weights/Slice_2SliceTinput_layer/sparse_input_layer/input_layer/C8_embedding/C8_embedding_weights/Shape_1Zinput_layer/sparse_input_layer/input_layer/C8_embedding/C8_embedding_weights/Slice_2/beginYinput_layer/sparse_input_layer/input_layer/C8_embedding/C8_embedding_weights/Slice_2/size*
T0*
_output_shapes
:*
Index0
Ъ
Xinput_layer/sparse_input_layer/input_layer/C8_embedding/C8_embedding_weights/concat/axisConst*
dtype0*
value	B : *
_output_shapes
: 
У
Sinput_layer/sparse_input_layer/input_layer/C8_embedding/C8_embedding_weights/concatConcatV2Tinput_layer/sparse_input_layer/input_layer/C8_embedding/C8_embedding_weights/Slice_1Tinput_layer/sparse_input_layer/input_layer/C8_embedding/C8_embedding_weights/Slice_2Xinput_layer/sparse_input_layer/input_layer/C8_embedding/C8_embedding_weights/concat/axis*
T0*
_output_shapes
:*
N
ґ
Vinput_layer/sparse_input_layer/input_layer/C8_embedding/C8_embedding_weights/Reshape_2ReshapeLinput_layer/sparse_input_layer/input_layer/C8_embedding/C8_embedding_weightsSinput_layer/sparse_input_layer/input_layer/C8_embedding/C8_embedding_weights/concat*
T0*'
_output_shapes
:€€€€€€€€€
√
=input_layer/sparse_input_layer/input_layer/C8_embedding/ShapeShapeVinput_layer/sparse_input_layer/input_layer/C8_embedding/C8_embedding_weights/Reshape_2*
_output_shapes
:*
T0
Х
Kinput_layer/sparse_input_layer/input_layer/C8_embedding/strided_slice/stackConst*
valueB: *
_output_shapes
:*
dtype0
Ч
Minput_layer/sparse_input_layer/input_layer/C8_embedding/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
Ч
Minput_layer/sparse_input_layer/input_layer/C8_embedding/strided_slice/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
≈
Einput_layer/sparse_input_layer/input_layer/C8_embedding/strided_sliceStridedSlice=input_layer/sparse_input_layer/input_layer/C8_embedding/ShapeKinput_layer/sparse_input_layer/input_layer/C8_embedding/strided_slice/stackMinput_layer/sparse_input_layer/input_layer/C8_embedding/strided_slice/stack_1Minput_layer/sparse_input_layer/input_layer/C8_embedding/strided_slice/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
Й
Ginput_layer/sparse_input_layer/input_layer/C8_embedding/Reshape/shape/1Const*
dtype0*
_output_shapes
: *
value	B :
Л
Einput_layer/sparse_input_layer/input_layer/C8_embedding/Reshape/shapePackEinput_layer/sparse_input_layer/input_layer/C8_embedding/strided_sliceGinput_layer/sparse_input_layer/input_layer/C8_embedding/Reshape/shape/1*
_output_shapes
:*
T0*
N
Ы
?input_layer/sparse_input_layer/input_layer/C8_embedding/ReshapeReshapeVinput_layer/sparse_input_layer/input_layer/C8_embedding/C8_embedding_weights/Reshape_2Einput_layer/sparse_input_layer/input_layer/C8_embedding/Reshape/shape*
T0*'
_output_shapes
:€€€€€€€€€
С
Finput_layer/sparse_input_layer/input_layer/C9_embedding/ExpandDims/dimConst*
dtype0*
_output_shapes
: *
valueB :
€€€€€€€€€
Џ
Binput_layer/sparse_input_layer/input_layer/C9_embedding/ExpandDims
ExpandDimsPlaceholder_21Finput_layer/sparse_input_layer/input_layer/C9_embedding/ExpandDims/dim*
T0*'
_output_shapes
:€€€€€€€€€
Ч
Vinput_layer/sparse_input_layer/input_layer/C9_embedding/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
valueB B *
dtype0
™
Pinput_layer/sparse_input_layer/input_layer/C9_embedding/to_sparse_input/NotEqualNotEqualBinput_layer/sparse_input_layer/input_layer/C9_embedding/ExpandDimsVinput_layer/sparse_input_layer/input_layer/C9_embedding/to_sparse_input/ignore_value/x*
T0*'
_output_shapes
:€€€€€€€€€
”
Oinput_layer/sparse_input_layer/input_layer/C9_embedding/to_sparse_input/indicesWherePinput_layer/sparse_input_layer/input_layer/C9_embedding/to_sparse_input/NotEqual*'
_output_shapes
:€€€€€€€€€
≥
Ninput_layer/sparse_input_layer/input_layer/C9_embedding/to_sparse_input/valuesGatherNdBinput_layer/sparse_input_layer/input_layer/C9_embedding/ExpandDimsOinput_layer/sparse_input_layer/input_layer/C9_embedding/to_sparse_input/indices*
Tparams0*
Tindices0	*#
_output_shapes
:€€€€€€€€€
’
Sinput_layer/sparse_input_layer/input_layer/C9_embedding/to_sparse_input/dense_shapeShapeBinput_layer/sparse_input_layer/input_layer/C9_embedding/ExpandDims*
out_type0	*
_output_shapes
:*
T0
б
>input_layer/sparse_input_layer/input_layer/C9_embedding/lookupStringToHashBucketFastNinput_layer/sparse_input_layer/input_layer/C9_embedding/to_sparse_input/values*#
_output_shapes
:€€€€€€€€€*
num_bucketsРN
Ы
linput_layer/sparse_input_layer/input_layer/C9_embedding/embedding_weights/Initializer/truncated_normal/shapeConst*
dtype0*
valueB"'     *
_output_shapes
:*\
_classR
PNloc:@input_layer/sparse_input_layer/input_layer/C9_embedding/embedding_weights
О
kinput_layer/sparse_input_layer/input_layer/C9_embedding/embedding_weights/Initializer/truncated_normal/meanConst*
dtype0*
_output_shapes
: *\
_classR
PNloc:@input_layer/sparse_input_layer/input_layer/C9_embedding/embedding_weights*
valueB
 *    
Р
minput_layer/sparse_input_layer/input_layer/C9_embedding/embedding_weights/Initializer/truncated_normal/stddevConst*
valueB
 *  А>*\
_classR
PNloc:@input_layer/sparse_input_layer/input_layer/C9_embedding/embedding_weights*
dtype0*
_output_shapes
: 
М
vinput_layer/sparse_input_layer/input_layer/C9_embedding/embedding_weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormallinput_layer/sparse_input_layer/input_layer/C9_embedding/embedding_weights/Initializer/truncated_normal/shape*\
_classR
PNloc:@input_layer/sparse_input_layer/input_layer/C9_embedding/embedding_weights*
_output_shapes
:	РN*
dtype0*
T0
а
jinput_layer/sparse_input_layer/input_layer/C9_embedding/embedding_weights/Initializer/truncated_normal/mulMulvinput_layer/sparse_input_layer/input_layer/C9_embedding/embedding_weights/Initializer/truncated_normal/TruncatedNormalminput_layer/sparse_input_layer/input_layer/C9_embedding/embedding_weights/Initializer/truncated_normal/stddev*\
_classR
PNloc:@input_layer/sparse_input_layer/input_layer/C9_embedding/embedding_weights*
T0*
_output_shapes
:	РN
ќ
finput_layer/sparse_input_layer/input_layer/C9_embedding/embedding_weights/Initializer/truncated_normalAddjinput_layer/sparse_input_layer/input_layer/C9_embedding/embedding_weights/Initializer/truncated_normal/mulkinput_layer/sparse_input_layer/input_layer/C9_embedding/embedding_weights/Initializer/truncated_normal/mean*
_output_shapes
:	РN*\
_classR
PNloc:@input_layer/sparse_input_layer/input_layer/C9_embedding/embedding_weights*
T0
щ
Iinput_layer/sparse_input_layer/input_layer/C9_embedding/embedding_weights
VariableV2*\
_classR
PNloc:@input_layer/sparse_input_layer/input_layer/C9_embedding/embedding_weights*
shape:	РN*
_output_shapes
:	РN*
dtype0
Х
Pinput_layer/sparse_input_layer/input_layer/C9_embedding/embedding_weights/AssignAssignIinput_layer/sparse_input_layer/input_layer/C9_embedding/embedding_weightsfinput_layer/sparse_input_layer/input_layer/C9_embedding/embedding_weights/Initializer/truncated_normal*
_output_shapes
:	РN*
T0*\
_classR
PNloc:@input_layer/sparse_input_layer/input_layer/C9_embedding/embedding_weights
≠
Ninput_layer/sparse_input_layer/input_layer/C9_embedding/embedding_weights/readIdentityIinput_layer/sparse_input_layer/input_layer/C9_embedding/embedding_weights*
_output_shapes
:	РN*
T0*\
_classR
PNloc:@input_layer/sparse_input_layer/input_layer/C9_embedding/embedding_weights
Ґ
Xinput_layer/sparse_input_layer/input_layer/C9_embedding/C9_embedding_weights/Slice/beginConst*
_output_shapes
:*
valueB: *
dtype0
°
Winput_layer/sparse_input_layer/input_layer/C9_embedding/C9_embedding_weights/Slice/sizeConst*
valueB:*
_output_shapes
:*
dtype0
Х
Rinput_layer/sparse_input_layer/input_layer/C9_embedding/C9_embedding_weights/SliceSliceSinput_layer/sparse_input_layer/input_layer/C9_embedding/to_sparse_input/dense_shapeXinput_layer/sparse_input_layer/input_layer/C9_embedding/C9_embedding_weights/Slice/beginWinput_layer/sparse_input_layer/input_layer/C9_embedding/C9_embedding_weights/Slice/size*
_output_shapes
:*
T0	*
Index0
Ь
Rinput_layer/sparse_input_layer/input_layer/C9_embedding/C9_embedding_weights/ConstConst*
_output_shapes
:*
valueB: *
dtype0
Ґ
Qinput_layer/sparse_input_layer/input_layer/C9_embedding/C9_embedding_weights/ProdProdRinput_layer/sparse_input_layer/input_layer/C9_embedding/C9_embedding_weights/SliceRinput_layer/sparse_input_layer/input_layer/C9_embedding/C9_embedding_weights/Const*
T0	*
_output_shapes
: 
Я
]input_layer/sparse_input_layer/input_layer/C9_embedding/C9_embedding_weights/GatherV2/indicesConst*
value	B :*
dtype0*
_output_shapes
: 
Ь
Zinput_layer/sparse_input_layer/input_layer/C9_embedding/C9_embedding_weights/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 
µ
Uinput_layer/sparse_input_layer/input_layer/C9_embedding/C9_embedding_weights/GatherV2GatherV2Sinput_layer/sparse_input_layer/input_layer/C9_embedding/to_sparse_input/dense_shape]input_layer/sparse_input_layer/input_layer/C9_embedding/C9_embedding_weights/GatherV2/indicesZinput_layer/sparse_input_layer/input_layer/C9_embedding/C9_embedding_weights/GatherV2/axis*
Taxis0*
Tparams0	*
Tindices0*
_output_shapes
: 
≥
Sinput_layer/sparse_input_layer/input_layer/C9_embedding/C9_embedding_weights/Cast/xPackQinput_layer/sparse_input_layer/input_layer/C9_embedding/C9_embedding_weights/ProdUinput_layer/sparse_input_layer/input_layer/C9_embedding/C9_embedding_weights/GatherV2*
N*
T0	*
_output_shapes
:
Х
Zinput_layer/sparse_input_layer/input_layer/C9_embedding/C9_embedding_weights/SparseReshapeSparseReshapeOinput_layer/sparse_input_layer/input_layer/C9_embedding/to_sparse_input/indicesSinput_layer/sparse_input_layer/input_layer/C9_embedding/to_sparse_input/dense_shapeSinput_layer/sparse_input_layer/input_layer/C9_embedding/C9_embedding_weights/Cast/x*-
_output_shapes
:€€€€€€€€€:
Ё
cinput_layer/sparse_input_layer/input_layer/C9_embedding/C9_embedding_weights/SparseReshape/IdentityIdentity>input_layer/sparse_input_layer/input_layer/C9_embedding/lookup*
T0	*#
_output_shapes
:€€€€€€€€€
Э
[input_layer/sparse_input_layer/input_layer/C9_embedding/C9_embedding_weights/GreaterEqual/yConst*
dtype0	*
_output_shapes
: *
value	B	 R 
ў
Yinput_layer/sparse_input_layer/input_layer/C9_embedding/C9_embedding_weights/GreaterEqualGreaterEqualcinput_layer/sparse_input_layer/input_layer/C9_embedding/C9_embedding_weights/SparseReshape/Identity[input_layer/sparse_input_layer/input_layer/C9_embedding/C9_embedding_weights/GreaterEqual/y*
T0	*#
_output_shapes
:€€€€€€€€€
я
Rinput_layer/sparse_input_layer/input_layer/C9_embedding/C9_embedding_weights/WhereWhereYinput_layer/sparse_input_layer/input_layer/C9_embedding/C9_embedding_weights/GreaterEqual*'
_output_shapes
:€€€€€€€€€
≠
Zinput_layer/sparse_input_layer/input_layer/C9_embedding/C9_embedding_weights/Reshape/shapeConst*
_output_shapes
:*
valueB:
€€€€€€€€€*
dtype0
љ
Tinput_layer/sparse_input_layer/input_layer/C9_embedding/C9_embedding_weights/ReshapeReshapeRinput_layer/sparse_input_layer/input_layer/C9_embedding/C9_embedding_weights/WhereZinput_layer/sparse_input_layer/input_layer/C9_embedding/C9_embedding_weights/Reshape/shape*
T0	*#
_output_shapes
:€€€€€€€€€
Ю
\input_layer/sparse_input_layer/input_layer/C9_embedding/C9_embedding_weights/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
»
Winput_layer/sparse_input_layer/input_layer/C9_embedding/C9_embedding_weights/GatherV2_1GatherV2Zinput_layer/sparse_input_layer/input_layer/C9_embedding/C9_embedding_weights/SparseReshapeTinput_layer/sparse_input_layer/input_layer/C9_embedding/C9_embedding_weights/Reshape\input_layer/sparse_input_layer/input_layer/C9_embedding/C9_embedding_weights/GatherV2_1/axis*'
_output_shapes
:€€€€€€€€€*
Tparams0	*
Tindices0	*
Taxis0
Ю
\input_layer/sparse_input_layer/input_layer/C9_embedding/C9_embedding_weights/GatherV2_2/axisConst*
value	B : *
dtype0*
_output_shapes
: 
Ќ
Winput_layer/sparse_input_layer/input_layer/C9_embedding/C9_embedding_weights/GatherV2_2GatherV2cinput_layer/sparse_input_layer/input_layer/C9_embedding/C9_embedding_weights/SparseReshape/IdentityTinput_layer/sparse_input_layer/input_layer/C9_embedding/C9_embedding_weights/Reshape\input_layer/sparse_input_layer/input_layer/C9_embedding/C9_embedding_weights/GatherV2_2/axis*
Tindices0	*
Tparams0	*#
_output_shapes
:€€€€€€€€€*
Taxis0
д
Uinput_layer/sparse_input_layer/input_layer/C9_embedding/C9_embedding_weights/IdentityIdentity\input_layer/sparse_input_layer/input_layer/C9_embedding/C9_embedding_weights/SparseReshape:1*
_output_shapes
:*
T0	
®
finput_layer/sparse_input_layer/input_layer/C9_embedding/C9_embedding_weights/SparseFillEmptyRows/ConstConst*
_output_shapes
: *
value	B	 R *
dtype0	
џ
tinput_layer/sparse_input_layer/input_layer/C9_embedding/C9_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowsSparseFillEmptyRowsWinput_layer/sparse_input_layer/input_layer/C9_embedding/C9_embedding_weights/GatherV2_1Winput_layer/sparse_input_layer/input_layer/C9_embedding/C9_embedding_weights/GatherV2_2Uinput_layer/sparse_input_layer/input_layer/C9_embedding/C9_embedding_weights/Identityfinput_layer/sparse_input_layer/input_layer/C9_embedding/C9_embedding_weights/SparseFillEmptyRows/Const*
T0	*T
_output_shapesB
@:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€
…
xinput_layer/sparse_input_layer/input_layer/C9_embedding/C9_embedding_weights/embedding_lookup_sparse/strided_slice/stackConst*
valueB"        *
_output_shapes
:*
dtype0
Ћ
zinput_layer/sparse_input_layer/input_layer/C9_embedding/C9_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1Const*
valueB"       *
_output_shapes
:*
dtype0
Ћ
zinput_layer/sparse_input_layer/input_layer/C9_embedding/C9_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB"      
я
rinput_layer/sparse_input_layer/input_layer/C9_embedding/C9_embedding_weights/embedding_lookup_sparse/strided_sliceStridedSlicetinput_layer/sparse_input_layer/input_layer/C9_embedding/C9_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowsxinput_layer/sparse_input_layer/input_layer/C9_embedding/C9_embedding_weights/embedding_lookup_sparse/strided_slice/stackzinput_layer/sparse_input_layer/input_layer/C9_embedding/C9_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1zinput_layer/sparse_input_layer/input_layer/C9_embedding/C9_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2*
Index0*
shrink_axis_mask*#
_output_shapes
:€€€€€€€€€*
end_mask*
T0	*

begin_mask
Ґ
iinput_layer/sparse_input_layer/input_layer/C9_embedding/C9_embedding_weights/embedding_lookup_sparse/CastCastrinput_layer/sparse_input_layer/input_layer/C9_embedding/C9_embedding_weights/embedding_lookup_sparse/strided_slice*

SrcT0	*

DstT0*#
_output_shapes
:€€€€€€€€€
™
kinput_layer/sparse_input_layer/input_layer/C9_embedding/C9_embedding_weights/embedding_lookup_sparse/UniqueUniquevinput_layer/sparse_input_layer/input_layer/C9_embedding/C9_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€*
T0	
Ъ
zinput_layer/sparse_input_layer/input_layer/C9_embedding/C9_embedding_weights/embedding_lookup_sparse/embedding_lookup/axisConst*
_output_shapes
: *\
_classR
PNloc:@input_layer/sparse_input_layer/input_layer/C9_embedding/embedding_weights*
dtype0*
value	B : 
н
uinput_layer/sparse_input_layer/input_layer/C9_embedding/C9_embedding_weights/embedding_lookup_sparse/embedding_lookupGatherV2Ninput_layer/sparse_input_layer/input_layer/C9_embedding/embedding_weights/readkinput_layer/sparse_input_layer/input_layer/C9_embedding/C9_embedding_weights/embedding_lookup_sparse/Uniquezinput_layer/sparse_input_layer/input_layer/C9_embedding/C9_embedding_weights/embedding_lookup_sparse/embedding_lookup/axis*'
_output_shapes
:€€€€€€€€€*
Taxis0*
Tindices0	*\
_classR
PNloc:@input_layer/sparse_input_layer/input_layer/C9_embedding/embedding_weights*
Tparams0
≥
~input_layer/sparse_input_layer/input_layer/C9_embedding/C9_embedding_weights/embedding_lookup_sparse/embedding_lookup/IdentityIdentityuinput_layer/sparse_input_layer/input_layer/C9_embedding/C9_embedding_weights/embedding_lookup_sparse/embedding_lookup*
T0*'
_output_shapes
:€€€€€€€€€
Е
dinput_layer/sparse_input_layer/input_layer/C9_embedding/C9_embedding_weights/embedding_lookup_sparseSparseSegmentMean~input_layer/sparse_input_layer/input_layer/C9_embedding/C9_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identityminput_layer/sparse_input_layer/input_layer/C9_embedding/C9_embedding_weights/embedding_lookup_sparse/Unique:1iinput_layer/sparse_input_layer/input_layer/C9_embedding/C9_embedding_weights/embedding_lookup_sparse/Cast*'
_output_shapes
:€€€€€€€€€*
T0
≠
\input_layer/sparse_input_layer/input_layer/C9_embedding/C9_embedding_weights/Reshape_1/shapeConst*
valueB"€€€€   *
_output_shapes
:*
dtype0
й
Vinput_layer/sparse_input_layer/input_layer/C9_embedding/C9_embedding_weights/Reshape_1Reshapevinput_layer/sparse_input_layer/input_layer/C9_embedding/C9_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:2\input_layer/sparse_input_layer/input_layer/C9_embedding/C9_embedding_weights/Reshape_1/shape*
T0
*'
_output_shapes
:€€€€€€€€€
ж
Rinput_layer/sparse_input_layer/input_layer/C9_embedding/C9_embedding_weights/ShapeShapedinput_layer/sparse_input_layer/input_layer/C9_embedding/C9_embedding_weights/embedding_lookup_sparse*
_output_shapes
:*
T0
™
`input_layer/sparse_input_layer/input_layer/C9_embedding/C9_embedding_weights/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB:
ђ
binput_layer/sparse_input_layer/input_layer/C9_embedding/C9_embedding_weights/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ђ
binput_layer/sparse_input_layer/input_layer/C9_embedding/C9_embedding_weights/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
Ѓ
Zinput_layer/sparse_input_layer/input_layer/C9_embedding/C9_embedding_weights/strided_sliceStridedSliceRinput_layer/sparse_input_layer/input_layer/C9_embedding/C9_embedding_weights/Shape`input_layer/sparse_input_layer/input_layer/C9_embedding/C9_embedding_weights/strided_slice/stackbinput_layer/sparse_input_layer/input_layer/C9_embedding/C9_embedding_weights/strided_slice/stack_1binput_layer/sparse_input_layer/input_layer/C9_embedding/C9_embedding_weights/strided_slice/stack_2*
Index0*
shrink_axis_mask*
T0*
_output_shapes
: 
Ц
Tinput_layer/sparse_input_layer/input_layer/C9_embedding/C9_embedding_weights/stack/0Const*
value	B :*
_output_shapes
: *
dtype0
Ї
Rinput_layer/sparse_input_layer/input_layer/C9_embedding/C9_embedding_weights/stackPackTinput_layer/sparse_input_layer/input_layer/C9_embedding/C9_embedding_weights/stack/0Zinput_layer/sparse_input_layer/input_layer/C9_embedding/C9_embedding_weights/strided_slice*
T0*
N*
_output_shapes
:
ј
Qinput_layer/sparse_input_layer/input_layer/C9_embedding/C9_embedding_weights/TileTileVinput_layer/sparse_input_layer/input_layer/C9_embedding/C9_embedding_weights/Reshape_1Rinput_layer/sparse_input_layer/input_layer/C9_embedding/C9_embedding_weights/stack*0
_output_shapes
:€€€€€€€€€€€€€€€€€€*
T0

ь
Winput_layer/sparse_input_layer/input_layer/C9_embedding/C9_embedding_weights/zeros_like	ZerosLikedinput_layer/sparse_input_layer/input_layer/C9_embedding/C9_embedding_weights/embedding_lookup_sparse*'
_output_shapes
:€€€€€€€€€*
T0
Ъ
Linput_layer/sparse_input_layer/input_layer/C9_embedding/C9_embedding_weightsSelectQinput_layer/sparse_input_layer/input_layer/C9_embedding/C9_embedding_weights/TileWinput_layer/sparse_input_layer/input_layer/C9_embedding/C9_embedding_weights/zeros_likedinput_layer/sparse_input_layer/input_layer/C9_embedding/C9_embedding_weights/embedding_lookup_sparse*
T0*'
_output_shapes
:€€€€€€€€€
д
Sinput_layer/sparse_input_layer/input_layer/C9_embedding/C9_embedding_weights/Cast_1CastSinput_layer/sparse_input_layer/input_layer/C9_embedding/to_sparse_input/dense_shape*

SrcT0	*
_output_shapes
:*

DstT0
§
Zinput_layer/sparse_input_layer/input_layer/C9_embedding/C9_embedding_weights/Slice_1/beginConst*
dtype0*
valueB: *
_output_shapes
:
£
Yinput_layer/sparse_input_layer/input_layer/C9_embedding/C9_embedding_weights/Slice_1/sizeConst*
dtype0*
_output_shapes
:*
valueB:
Ы
Tinput_layer/sparse_input_layer/input_layer/C9_embedding/C9_embedding_weights/Slice_1SliceSinput_layer/sparse_input_layer/input_layer/C9_embedding/C9_embedding_weights/Cast_1Zinput_layer/sparse_input_layer/input_layer/C9_embedding/C9_embedding_weights/Slice_1/beginYinput_layer/sparse_input_layer/input_layer/C9_embedding/C9_embedding_weights/Slice_1/size*
T0*
Index0*
_output_shapes
:
–
Tinput_layer/sparse_input_layer/input_layer/C9_embedding/C9_embedding_weights/Shape_1ShapeLinput_layer/sparse_input_layer/input_layer/C9_embedding/C9_embedding_weights*
_output_shapes
:*
T0
§
Zinput_layer/sparse_input_layer/input_layer/C9_embedding/C9_embedding_weights/Slice_2/beginConst*
_output_shapes
:*
valueB:*
dtype0
ђ
Yinput_layer/sparse_input_layer/input_layer/C9_embedding/C9_embedding_weights/Slice_2/sizeConst*
_output_shapes
:*
valueB:
€€€€€€€€€*
dtype0
Ь
Tinput_layer/sparse_input_layer/input_layer/C9_embedding/C9_embedding_weights/Slice_2SliceTinput_layer/sparse_input_layer/input_layer/C9_embedding/C9_embedding_weights/Shape_1Zinput_layer/sparse_input_layer/input_layer/C9_embedding/C9_embedding_weights/Slice_2/beginYinput_layer/sparse_input_layer/input_layer/C9_embedding/C9_embedding_weights/Slice_2/size*
T0*
Index0*
_output_shapes
:
Ъ
Xinput_layer/sparse_input_layer/input_layer/C9_embedding/C9_embedding_weights/concat/axisConst*
dtype0*
value	B : *
_output_shapes
: 
У
Sinput_layer/sparse_input_layer/input_layer/C9_embedding/C9_embedding_weights/concatConcatV2Tinput_layer/sparse_input_layer/input_layer/C9_embedding/C9_embedding_weights/Slice_1Tinput_layer/sparse_input_layer/input_layer/C9_embedding/C9_embedding_weights/Slice_2Xinput_layer/sparse_input_layer/input_layer/C9_embedding/C9_embedding_weights/concat/axis*
N*
_output_shapes
:*
T0
ґ
Vinput_layer/sparse_input_layer/input_layer/C9_embedding/C9_embedding_weights/Reshape_2ReshapeLinput_layer/sparse_input_layer/input_layer/C9_embedding/C9_embedding_weightsSinput_layer/sparse_input_layer/input_layer/C9_embedding/C9_embedding_weights/concat*'
_output_shapes
:€€€€€€€€€*
T0
√
=input_layer/sparse_input_layer/input_layer/C9_embedding/ShapeShapeVinput_layer/sparse_input_layer/input_layer/C9_embedding/C9_embedding_weights/Reshape_2*
T0*
_output_shapes
:
Х
Kinput_layer/sparse_input_layer/input_layer/C9_embedding/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB: 
Ч
Minput_layer/sparse_input_layer/input_layer/C9_embedding/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
Ч
Minput_layer/sparse_input_layer/input_layer/C9_embedding/strided_slice/stack_2Const*
valueB:*
_output_shapes
:*
dtype0
≈
Einput_layer/sparse_input_layer/input_layer/C9_embedding/strided_sliceStridedSlice=input_layer/sparse_input_layer/input_layer/C9_embedding/ShapeKinput_layer/sparse_input_layer/input_layer/C9_embedding/strided_slice/stackMinput_layer/sparse_input_layer/input_layer/C9_embedding/strided_slice/stack_1Minput_layer/sparse_input_layer/input_layer/C9_embedding/strided_slice/stack_2*
T0*
_output_shapes
: *
Index0*
shrink_axis_mask
Й
Ginput_layer/sparse_input_layer/input_layer/C9_embedding/Reshape/shape/1Const*
value	B :*
dtype0*
_output_shapes
: 
Л
Einput_layer/sparse_input_layer/input_layer/C9_embedding/Reshape/shapePackEinput_layer/sparse_input_layer/input_layer/C9_embedding/strided_sliceGinput_layer/sparse_input_layer/input_layer/C9_embedding/Reshape/shape/1*
N*
T0*
_output_shapes
:
Ы
?input_layer/sparse_input_layer/input_layer/C9_embedding/ReshapeReshapeVinput_layer/sparse_input_layer/input_layer/C9_embedding/C9_embedding_weights/Reshape_2Einput_layer/sparse_input_layer/input_layer/C9_embedding/Reshape/shape*
T0*'
_output_shapes
:€€€€€€€€€
x
6input_layer/sparse_input_layer/input_layer/concat/axisConst*
value	B :*
_output_shapes
: *
dtype0
№
1input_layer/sparse_input_layer/input_layer/concatConcatV2@input_layer/sparse_input_layer/input_layer/C10_embedding/Reshape@input_layer/sparse_input_layer/input_layer/C11_embedding/Reshape@input_layer/sparse_input_layer/input_layer/C12_embedding/Reshape@input_layer/sparse_input_layer/input_layer/C13_embedding/Reshape@input_layer/sparse_input_layer/input_layer/C14_embedding/Reshape@input_layer/sparse_input_layer/input_layer/C15_embedding/Reshape@input_layer/sparse_input_layer/input_layer/C16_embedding/Reshape@input_layer/sparse_input_layer/input_layer/C17_embedding/Reshape@input_layer/sparse_input_layer/input_layer/C18_embedding/Reshape@input_layer/sparse_input_layer/input_layer/C19_embedding/Reshape?input_layer/sparse_input_layer/input_layer/C1_embedding/Reshape@input_layer/sparse_input_layer/input_layer/C20_embedding/Reshape@input_layer/sparse_input_layer/input_layer/C21_embedding/Reshape@input_layer/sparse_input_layer/input_layer/C22_embedding/Reshape@input_layer/sparse_input_layer/input_layer/C23_embedding/Reshape@input_layer/sparse_input_layer/input_layer/C24_embedding/Reshape@input_layer/sparse_input_layer/input_layer/C25_embedding/Reshape@input_layer/sparse_input_layer/input_layer/C26_embedding/Reshape?input_layer/sparse_input_layer/input_layer/C2_embedding/Reshape?input_layer/sparse_input_layer/input_layer/C3_embedding/Reshape?input_layer/sparse_input_layer/input_layer/C4_embedding/Reshape?input_layer/sparse_input_layer/input_layer/C5_embedding/Reshape?input_layer/sparse_input_layer/input_layer/C6_embedding/Reshape?input_layer/sparse_input_layer/input_layer/C7_embedding/Reshape?input_layer/sparse_input_layer/input_layer/C8_embedding/Reshape?input_layer/sparse_input_layer/input_layer/C9_embedding/Reshape6input_layer/sparse_input_layer/input_layer/concat/axis*
N*(
_output_shapes
:€€€€€€€€€†*
T0
й
Rmlp_bot_layer/mlp_bot_hiddenlayer_0/kernel/part_0/Initializer/random_uniform/shapeConst*D
_class:
86loc:@mlp_bot_layer/mlp_bot_hiddenlayer_0/kernel/part_0*
_output_shapes
:*
dtype0*
valueB"      
џ
Pmlp_bot_layer/mlp_bot_hiddenlayer_0/kernel/part_0/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *њрЏљ*D
_class:
86loc:@mlp_bot_layer/mlp_bot_hiddenlayer_0/kernel/part_0
џ
Pmlp_bot_layer/mlp_bot_hiddenlayer_0/kernel/part_0/Initializer/random_uniform/maxConst*
_output_shapes
: *D
_class:
86loc:@mlp_bot_layer/mlp_bot_hiddenlayer_0/kernel/part_0*
dtype0*
valueB
 *њрЏ=
Љ
Zmlp_bot_layer/mlp_bot_hiddenlayer_0/kernel/part_0/Initializer/random_uniform/RandomUniformRandomUniformRmlp_bot_layer/mlp_bot_hiddenlayer_0/kernel/part_0/Initializer/random_uniform/shape*
dtype0*D
_class:
86loc:@mlp_bot_layer/mlp_bot_hiddenlayer_0/kernel/part_0*
_output_shapes
:	А*
T0
в
Pmlp_bot_layer/mlp_bot_hiddenlayer_0/kernel/part_0/Initializer/random_uniform/subSubPmlp_bot_layer/mlp_bot_hiddenlayer_0/kernel/part_0/Initializer/random_uniform/maxPmlp_bot_layer/mlp_bot_hiddenlayer_0/kernel/part_0/Initializer/random_uniform/min*
T0*D
_class:
86loc:@mlp_bot_layer/mlp_bot_hiddenlayer_0/kernel/part_0*
_output_shapes
: 
х
Pmlp_bot_layer/mlp_bot_hiddenlayer_0/kernel/part_0/Initializer/random_uniform/mulMulZmlp_bot_layer/mlp_bot_hiddenlayer_0/kernel/part_0/Initializer/random_uniform/RandomUniformPmlp_bot_layer/mlp_bot_hiddenlayer_0/kernel/part_0/Initializer/random_uniform/sub*D
_class:
86loc:@mlp_bot_layer/mlp_bot_hiddenlayer_0/kernel/part_0*
_output_shapes
:	А*
T0
з
Lmlp_bot_layer/mlp_bot_hiddenlayer_0/kernel/part_0/Initializer/random_uniformAddPmlp_bot_layer/mlp_bot_hiddenlayer_0/kernel/part_0/Initializer/random_uniform/mulPmlp_bot_layer/mlp_bot_hiddenlayer_0/kernel/part_0/Initializer/random_uniform/min*D
_class:
86loc:@mlp_bot_layer/mlp_bot_hiddenlayer_0/kernel/part_0*
T0*
_output_shapes
:	А
…
1mlp_bot_layer/mlp_bot_hiddenlayer_0/kernel/part_0
VariableV2*D
_class:
86loc:@mlp_bot_layer/mlp_bot_hiddenlayer_0/kernel/part_0*
_output_shapes
:	А*
shape:	А*
dtype0
≥
8mlp_bot_layer/mlp_bot_hiddenlayer_0/kernel/part_0/AssignAssign1mlp_bot_layer/mlp_bot_hiddenlayer_0/kernel/part_0Lmlp_bot_layer/mlp_bot_hiddenlayer_0/kernel/part_0/Initializer/random_uniform*
T0*D
_class:
86loc:@mlp_bot_layer/mlp_bot_hiddenlayer_0/kernel/part_0*
_output_shapes
:	А
е
6mlp_bot_layer/mlp_bot_hiddenlayer_0/kernel/part_0/readIdentity1mlp_bot_layer/mlp_bot_hiddenlayer_0/kernel/part_0*
T0*
_output_shapes
:	А*D
_class:
86loc:@mlp_bot_layer/mlp_bot_hiddenlayer_0/kernel/part_0
‘
Amlp_bot_layer/mlp_bot_hiddenlayer_0/bias/part_0/Initializer/zerosConst*
dtype0*
valueBА*    *B
_class8
64loc:@mlp_bot_layer/mlp_bot_hiddenlayer_0/bias/part_0*
_output_shapes	
:А
љ
/mlp_bot_layer/mlp_bot_hiddenlayer_0/bias/part_0
VariableV2*
_output_shapes	
:А*
shape:А*
dtype0*B
_class8
64loc:@mlp_bot_layer/mlp_bot_hiddenlayer_0/bias/part_0
Ю
6mlp_bot_layer/mlp_bot_hiddenlayer_0/bias/part_0/AssignAssign/mlp_bot_layer/mlp_bot_hiddenlayer_0/bias/part_0Amlp_bot_layer/mlp_bot_hiddenlayer_0/bias/part_0/Initializer/zeros*B
_class8
64loc:@mlp_bot_layer/mlp_bot_hiddenlayer_0/bias/part_0*
T0*
_output_shapes	
:А
џ
4mlp_bot_layer/mlp_bot_hiddenlayer_0/bias/part_0/readIdentity/mlp_bot_layer/mlp_bot_hiddenlayer_0/bias/part_0*
T0*B
_class8
64loc:@mlp_bot_layer/mlp_bot_hiddenlayer_0/bias/part_0*
_output_shapes	
:А
Ш
*mlp_bot_layer/mlp_bot_hiddenlayer_0/kernelIdentity6mlp_bot_layer/mlp_bot_hiddenlayer_0/kernel/part_0/read*
_output_shapes
:	А*
T0
≈
*mlp_bot_layer/mlp_bot_hiddenlayer_0/MatMulMatMul0input_layer/dense_input_layer/input_layer/concat*mlp_bot_layer/mlp_bot_hiddenlayer_0/kernel*(
_output_shapes
:€€€€€€€€€А*
T0
Р
(mlp_bot_layer/mlp_bot_hiddenlayer_0/biasIdentity4mlp_bot_layer/mlp_bot_hiddenlayer_0/bias/part_0/read*
_output_shapes	
:А*
T0
њ
+mlp_bot_layer/mlp_bot_hiddenlayer_0/BiasAddBiasAdd*mlp_bot_layer/mlp_bot_hiddenlayer_0/MatMul(mlp_bot_layer/mlp_bot_hiddenlayer_0/bias*(
_output_shapes
:€€€€€€€€€А*
T0
Р
(mlp_bot_layer/mlp_bot_hiddenlayer_0/ReluRelu+mlp_bot_layer/mlp_bot_hiddenlayer_0/BiasAdd*(
_output_shapes
:€€€€€€€€€А*
T0
э
Umlp_bot_layer/mlp_bot_hiddenlayer_0/batch_normalization/gamma/part_0/Initializer/onesConst*
dtype0*W
_classM
KIloc:@mlp_bot_layer/mlp_bot_hiddenlayer_0/batch_normalization/gamma/part_0*
_output_shapes	
:А*
valueBА*  А?
з
Dmlp_bot_layer/mlp_bot_hiddenlayer_0/batch_normalization/gamma/part_0
VariableV2*W
_classM
KIloc:@mlp_bot_layer/mlp_bot_hiddenlayer_0/batch_normalization/gamma/part_0*
dtype0*
_output_shapes	
:А*
shape:А
с
Kmlp_bot_layer/mlp_bot_hiddenlayer_0/batch_normalization/gamma/part_0/AssignAssignDmlp_bot_layer/mlp_bot_hiddenlayer_0/batch_normalization/gamma/part_0Umlp_bot_layer/mlp_bot_hiddenlayer_0/batch_normalization/gamma/part_0/Initializer/ones*W
_classM
KIloc:@mlp_bot_layer/mlp_bot_hiddenlayer_0/batch_normalization/gamma/part_0*
T0*
_output_shapes	
:А
Ъ
Imlp_bot_layer/mlp_bot_hiddenlayer_0/batch_normalization/gamma/part_0/readIdentityDmlp_bot_layer/mlp_bot_hiddenlayer_0/batch_normalization/gamma/part_0*W
_classM
KIloc:@mlp_bot_layer/mlp_bot_hiddenlayer_0/batch_normalization/gamma/part_0*
T0*
_output_shapes	
:А
ь
Umlp_bot_layer/mlp_bot_hiddenlayer_0/batch_normalization/beta/part_0/Initializer/zerosConst*V
_classL
JHloc:@mlp_bot_layer/mlp_bot_hiddenlayer_0/batch_normalization/beta/part_0*
valueBА*    *
dtype0*
_output_shapes	
:А
е
Cmlp_bot_layer/mlp_bot_hiddenlayer_0/batch_normalization/beta/part_0
VariableV2*
_output_shapes	
:А*
dtype0*
shape:А*V
_classL
JHloc:@mlp_bot_layer/mlp_bot_hiddenlayer_0/batch_normalization/beta/part_0
о
Jmlp_bot_layer/mlp_bot_hiddenlayer_0/batch_normalization/beta/part_0/AssignAssignCmlp_bot_layer/mlp_bot_hiddenlayer_0/batch_normalization/beta/part_0Umlp_bot_layer/mlp_bot_hiddenlayer_0/batch_normalization/beta/part_0/Initializer/zeros*
_output_shapes	
:А*V
_classL
JHloc:@mlp_bot_layer/mlp_bot_hiddenlayer_0/batch_normalization/beta/part_0*
T0
Ч
Hmlp_bot_layer/mlp_bot_hiddenlayer_0/batch_normalization/beta/part_0/readIdentityCmlp_bot_layer/mlp_bot_hiddenlayer_0/batch_normalization/beta/part_0*
T0*V
_classL
JHloc:@mlp_bot_layer/mlp_bot_hiddenlayer_0/batch_normalization/beta/part_0*
_output_shapes	
:А
ь
Umlp_bot_layer/mlp_bot_hiddenlayer_0/batch_normalization/moving_mean/Initializer/zerosConst*V
_classL
JHloc:@mlp_bot_layer/mlp_bot_hiddenlayer_0/batch_normalization/moving_mean*
valueBА*    *
_output_shapes	
:А*
dtype0
е
Cmlp_bot_layer/mlp_bot_hiddenlayer_0/batch_normalization/moving_mean
VariableV2*V
_classL
JHloc:@mlp_bot_layer/mlp_bot_hiddenlayer_0/batch_normalization/moving_mean*
_output_shapes	
:А*
dtype0*
shape:А
о
Jmlp_bot_layer/mlp_bot_hiddenlayer_0/batch_normalization/moving_mean/AssignAssignCmlp_bot_layer/mlp_bot_hiddenlayer_0/batch_normalization/moving_meanUmlp_bot_layer/mlp_bot_hiddenlayer_0/batch_normalization/moving_mean/Initializer/zeros*
T0*
_output_shapes	
:А*V
_classL
JHloc:@mlp_bot_layer/mlp_bot_hiddenlayer_0/batch_normalization/moving_mean
Ч
Hmlp_bot_layer/mlp_bot_hiddenlayer_0/batch_normalization/moving_mean/readIdentityCmlp_bot_layer/mlp_bot_hiddenlayer_0/batch_normalization/moving_mean*
T0*V
_classL
JHloc:@mlp_bot_layer/mlp_bot_hiddenlayer_0/batch_normalization/moving_mean*
_output_shapes	
:А
Г
Xmlp_bot_layer/mlp_bot_hiddenlayer_0/batch_normalization/moving_variance/Initializer/onesConst*
valueBА*  А?*
_output_shapes	
:А*
dtype0*Z
_classP
NLloc:@mlp_bot_layer/mlp_bot_hiddenlayer_0/batch_normalization/moving_variance
н
Gmlp_bot_layer/mlp_bot_hiddenlayer_0/batch_normalization/moving_variance
VariableV2*Z
_classP
NLloc:@mlp_bot_layer/mlp_bot_hiddenlayer_0/batch_normalization/moving_variance*
dtype0*
_output_shapes	
:А*
shape:А
э
Nmlp_bot_layer/mlp_bot_hiddenlayer_0/batch_normalization/moving_variance/AssignAssignGmlp_bot_layer/mlp_bot_hiddenlayer_0/batch_normalization/moving_varianceXmlp_bot_layer/mlp_bot_hiddenlayer_0/batch_normalization/moving_variance/Initializer/ones*Z
_classP
NLloc:@mlp_bot_layer/mlp_bot_hiddenlayer_0/batch_normalization/moving_variance*
T0*
_output_shapes	
:А
£
Lmlp_bot_layer/mlp_bot_hiddenlayer_0/batch_normalization/moving_variance/readIdentityGmlp_bot_layer/mlp_bot_hiddenlayer_0/batch_normalization/moving_variance*Z
_classP
NLloc:@mlp_bot_layer/mlp_bot_hiddenlayer_0/batch_normalization/moving_variance*
T0*
_output_shapes	
:А
†
Vmlp_bot_layer/mlp_bot_hiddenlayer_0/batch_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
Й
Dmlp_bot_layer/mlp_bot_hiddenlayer_0/batch_normalization/moments/meanMean(mlp_bot_layer/mlp_bot_hiddenlayer_0/ReluVmlp_bot_layer/mlp_bot_hiddenlayer_0/batch_normalization/moments/mean/reduction_indices*
_output_shapes
:	А*
T0*
	keep_dims(
ћ
Lmlp_bot_layer/mlp_bot_hiddenlayer_0/batch_normalization/moments/StopGradientStopGradientDmlp_bot_layer/mlp_bot_hiddenlayer_0/batch_normalization/moments/mean*
T0*
_output_shapes
:	А
С
Qmlp_bot_layer/mlp_bot_hiddenlayer_0/batch_normalization/moments/SquaredDifferenceSquaredDifference(mlp_bot_layer/mlp_bot_hiddenlayer_0/ReluLmlp_bot_layer/mlp_bot_hiddenlayer_0/batch_normalization/moments/StopGradient*(
_output_shapes
:€€€€€€€€€А*
T0
§
Zmlp_bot_layer/mlp_bot_hiddenlayer_0/batch_normalization/moments/variance/reduction_indicesConst*
dtype0*
valueB: *
_output_shapes
:
Ї
Hmlp_bot_layer/mlp_bot_hiddenlayer_0/batch_normalization/moments/varianceMeanQmlp_bot_layer/mlp_bot_hiddenlayer_0/batch_normalization/moments/SquaredDifferenceZmlp_bot_layer/mlp_bot_hiddenlayer_0/batch_normalization/moments/variance/reduction_indices*
T0*
	keep_dims(*
_output_shapes
:	А
’
Gmlp_bot_layer/mlp_bot_hiddenlayer_0/batch_normalization/moments/SqueezeSqueezeDmlp_bot_layer/mlp_bot_hiddenlayer_0/batch_normalization/moments/mean*
_output_shapes	
:А*
squeeze_dims
 *
T0
џ
Imlp_bot_layer/mlp_bot_hiddenlayer_0/batch_normalization/moments/Squeeze_1SqueezeHmlp_bot_layer/mlp_bot_hiddenlayer_0/batch_normalization/moments/variance*
T0*
_output_shapes	
:А*
squeeze_dims
 
к
Mmlp_bot_layer/mlp_bot_hiddenlayer_0/batch_normalization/AssignMovingAvg/decayConst*V
_classL
JHloc:@mlp_bot_layer/mlp_bot_hiddenlayer_0/batch_normalization/moving_mean*
_output_shapes
: *
valueB
 *
„#<*
dtype0
г
Kmlp_bot_layer/mlp_bot_hiddenlayer_0/batch_normalization/AssignMovingAvg/subSubHmlp_bot_layer/mlp_bot_hiddenlayer_0/batch_normalization/moving_mean/readGmlp_bot_layer/mlp_bot_hiddenlayer_0/batch_normalization/moments/Squeeze*
_output_shapes	
:А*
T0*V
_classL
JHloc:@mlp_bot_layer/mlp_bot_hiddenlayer_0/batch_normalization/moving_mean
м
Kmlp_bot_layer/mlp_bot_hiddenlayer_0/batch_normalization/AssignMovingAvg/mulMulKmlp_bot_layer/mlp_bot_hiddenlayer_0/batch_normalization/AssignMovingAvg/subMmlp_bot_layer/mlp_bot_hiddenlayer_0/batch_normalization/AssignMovingAvg/decay*V
_classL
JHloc:@mlp_bot_layer/mlp_bot_hiddenlayer_0/batch_normalization/moving_mean*
_output_shapes	
:А*
T0
д
Gmlp_bot_layer/mlp_bot_hiddenlayer_0/batch_normalization/AssignMovingAvg	AssignSubCmlp_bot_layer/mlp_bot_hiddenlayer_0/batch_normalization/moving_meanKmlp_bot_layer/mlp_bot_hiddenlayer_0/batch_normalization/AssignMovingAvg/mul*
T0*
_output_shapes	
:А*V
_classL
JHloc:@mlp_bot_layer/mlp_bot_hiddenlayer_0/batch_normalization/moving_mean
р
Omlp_bot_layer/mlp_bot_hiddenlayer_0/batch_normalization/AssignMovingAvg_1/decayConst*
_output_shapes
: *Z
_classP
NLloc:@mlp_bot_layer/mlp_bot_hiddenlayer_0/batch_normalization/moving_variance*
dtype0*
valueB
 *
„#<
п
Mmlp_bot_layer/mlp_bot_hiddenlayer_0/batch_normalization/AssignMovingAvg_1/subSubLmlp_bot_layer/mlp_bot_hiddenlayer_0/batch_normalization/moving_variance/readImlp_bot_layer/mlp_bot_hiddenlayer_0/batch_normalization/moments/Squeeze_1*
_output_shapes	
:А*Z
_classP
NLloc:@mlp_bot_layer/mlp_bot_hiddenlayer_0/batch_normalization/moving_variance*
T0
ц
Mmlp_bot_layer/mlp_bot_hiddenlayer_0/batch_normalization/AssignMovingAvg_1/mulMulMmlp_bot_layer/mlp_bot_hiddenlayer_0/batch_normalization/AssignMovingAvg_1/subOmlp_bot_layer/mlp_bot_hiddenlayer_0/batch_normalization/AssignMovingAvg_1/decay*
T0*
_output_shapes	
:А*Z
_classP
NLloc:@mlp_bot_layer/mlp_bot_hiddenlayer_0/batch_normalization/moving_variance
р
Imlp_bot_layer/mlp_bot_hiddenlayer_0/batch_normalization/AssignMovingAvg_1	AssignSubGmlp_bot_layer/mlp_bot_hiddenlayer_0/batch_normalization/moving_varianceMmlp_bot_layer/mlp_bot_hiddenlayer_0/batch_normalization/AssignMovingAvg_1/mul*
_output_shapes	
:А*Z
_classP
NLloc:@mlp_bot_layer/mlp_bot_hiddenlayer_0/batch_normalization/moving_variance*
T0
Є
<mlp_bot_layer/mlp_bot_hiddenlayer_0/batch_normalization/betaIdentityHmlp_bot_layer/mlp_bot_hiddenlayer_0/batch_normalization/beta/part_0/read*
T0*
_output_shapes	
:А
Ї
=mlp_bot_layer/mlp_bot_hiddenlayer_0/batch_normalization/gammaIdentityImlp_bot_layer/mlp_bot_hiddenlayer_0/batch_normalization/gamma/part_0/read*
T0*
_output_shapes	
:А
М
Gmlp_bot_layer/mlp_bot_hiddenlayer_0/batch_normalization/batchnorm/add/yConst*
valueB
 *oГ:*
_output_shapes
: *
dtype0
И
Emlp_bot_layer/mlp_bot_hiddenlayer_0/batch_normalization/batchnorm/addAddV2Imlp_bot_layer/mlp_bot_hiddenlayer_0/batch_normalization/moments/Squeeze_1Gmlp_bot_layer/mlp_bot_hiddenlayer_0/batch_normalization/batchnorm/add/y*
T0*
_output_shapes	
:А
љ
Gmlp_bot_layer/mlp_bot_hiddenlayer_0/batch_normalization/batchnorm/RsqrtRsqrtEmlp_bot_layer/mlp_bot_hiddenlayer_0/batch_normalization/batchnorm/add*
_output_shapes	
:А*
T0
ъ
Emlp_bot_layer/mlp_bot_hiddenlayer_0/batch_normalization/batchnorm/mulMulGmlp_bot_layer/mlp_bot_hiddenlayer_0/batch_normalization/batchnorm/Rsqrt=mlp_bot_layer/mlp_bot_hiddenlayer_0/batch_normalization/gamma*
T0*
_output_shapes	
:А
т
Gmlp_bot_layer/mlp_bot_hiddenlayer_0/batch_normalization/batchnorm/mul_1Mul(mlp_bot_layer/mlp_bot_hiddenlayer_0/ReluEmlp_bot_layer/mlp_bot_hiddenlayer_0/batch_normalization/batchnorm/mul*(
_output_shapes
:€€€€€€€€€А*
T0
Д
Gmlp_bot_layer/mlp_bot_hiddenlayer_0/batch_normalization/batchnorm/mul_2MulGmlp_bot_layer/mlp_bot_hiddenlayer_0/batch_normalization/moments/SqueezeEmlp_bot_layer/mlp_bot_hiddenlayer_0/batch_normalization/batchnorm/mul*
T0*
_output_shapes	
:А
щ
Emlp_bot_layer/mlp_bot_hiddenlayer_0/batch_normalization/batchnorm/subSub<mlp_bot_layer/mlp_bot_hiddenlayer_0/batch_normalization/betaGmlp_bot_layer/mlp_bot_hiddenlayer_0/batch_normalization/batchnorm/mul_2*
T0*
_output_shapes	
:А
У
Gmlp_bot_layer/mlp_bot_hiddenlayer_0/batch_normalization/batchnorm/add_1AddV2Gmlp_bot_layer/mlp_bot_hiddenlayer_0/batch_normalization/batchnorm/mul_1Emlp_bot_layer/mlp_bot_hiddenlayer_0/batch_normalization/batchnorm/sub*
T0*(
_output_shapes
:€€€€€€€€€А
Є
6mlp_bot_layer/mlp_bot_hiddenlayer_0/zero_fraction/SizeSizeGmlp_bot_layer/mlp_bot_hiddenlayer_0/batch_normalization/batchnorm/add_1*
T0*
out_type0	*
_output_shapes
: 
Г
=mlp_bot_layer/mlp_bot_hiddenlayer_0/zero_fraction/LessEqual/yConst*
dtype0	*
_output_shapes
: *
valueB	 R€€€€
а
;mlp_bot_layer/mlp_bot_hiddenlayer_0/zero_fraction/LessEqual	LessEqual6mlp_bot_layer/mlp_bot_hiddenlayer_0/zero_fraction/Size=mlp_bot_layer/mlp_bot_hiddenlayer_0/zero_fraction/LessEqual/y*
_output_shapes
: *
T0	
д
=mlp_bot_layer/mlp_bot_hiddenlayer_0/zero_fraction/cond/SwitchSwitch;mlp_bot_layer/mlp_bot_hiddenlayer_0/zero_fraction/LessEqual;mlp_bot_layer/mlp_bot_hiddenlayer_0/zero_fraction/LessEqual*
_output_shapes
: : *
T0

≠
?mlp_bot_layer/mlp_bot_hiddenlayer_0/zero_fraction/cond/switch_tIdentity?mlp_bot_layer/mlp_bot_hiddenlayer_0/zero_fraction/cond/Switch:1*
T0
*
_output_shapes
: 
Ђ
?mlp_bot_layer/mlp_bot_hiddenlayer_0/zero_fraction/cond/switch_fIdentity=mlp_bot_layer/mlp_bot_hiddenlayer_0/zero_fraction/cond/Switch*
T0
*
_output_shapes
: 
®
>mlp_bot_layer/mlp_bot_hiddenlayer_0/zero_fraction/cond/pred_idIdentity;mlp_bot_layer/mlp_bot_hiddenlayer_0/zero_fraction/LessEqual*
_output_shapes
: *
T0

—
Jmlp_bot_layer/mlp_bot_hiddenlayer_0/zero_fraction/cond/count_nonzero/zerosConst@^mlp_bot_layer/mlp_bot_hiddenlayer_0/zero_fraction/cond/switch_t*
valueB
 *    *
dtype0*
_output_shapes
: 
∞
Mmlp_bot_layer/mlp_bot_hiddenlayer_0/zero_fraction/cond/count_nonzero/NotEqualNotEqualVmlp_bot_layer/mlp_bot_hiddenlayer_0/zero_fraction/cond/count_nonzero/NotEqual/Switch:1Jmlp_bot_layer/mlp_bot_hiddenlayer_0/zero_fraction/cond/count_nonzero/zeros*(
_output_shapes
:€€€€€€€€€А*
T0
К
Tmlp_bot_layer/mlp_bot_hiddenlayer_0/zero_fraction/cond/count_nonzero/NotEqual/SwitchSwitchGmlp_bot_layer/mlp_bot_hiddenlayer_0/batch_normalization/batchnorm/add_1>mlp_bot_layer/mlp_bot_hiddenlayer_0/zero_fraction/cond/pred_id*
T0*<
_output_shapes*
(:€€€€€€€€€А:€€€€€€€€€А*Z
_classP
NLloc:@mlp_bot_layer/mlp_bot_hiddenlayer_0/batch_normalization/batchnorm/add_1
в
Imlp_bot_layer/mlp_bot_hiddenlayer_0/zero_fraction/cond/count_nonzero/CastCastMmlp_bot_layer/mlp_bot_hiddenlayer_0/zero_fraction/cond/count_nonzero/NotEqual*

SrcT0
*(
_output_shapes
:€€€€€€€€€А*

DstT0
Ё
Jmlp_bot_layer/mlp_bot_hiddenlayer_0/zero_fraction/cond/count_nonzero/ConstConst@^mlp_bot_layer/mlp_bot_hiddenlayer_0/zero_fraction/cond/switch_t*
dtype0*
_output_shapes
:*
valueB"       
С
Rmlp_bot_layer/mlp_bot_hiddenlayer_0/zero_fraction/cond/count_nonzero/nonzero_countSumImlp_bot_layer/mlp_bot_hiddenlayer_0/zero_fraction/cond/count_nonzero/CastJmlp_bot_layer/mlp_bot_hiddenlayer_0/zero_fraction/cond/count_nonzero/Const*
T0*
_output_shapes
: 
«
;mlp_bot_layer/mlp_bot_hiddenlayer_0/zero_fraction/cond/CastCastRmlp_bot_layer/mlp_bot_hiddenlayer_0/zero_fraction/cond/count_nonzero/nonzero_count*

SrcT0*

DstT0	*
_output_shapes
: 
”
Lmlp_bot_layer/mlp_bot_hiddenlayer_0/zero_fraction/cond/count_nonzero_1/zerosConst@^mlp_bot_layer/mlp_bot_hiddenlayer_0/zero_fraction/cond/switch_f*
valueB
 *    *
_output_shapes
: *
dtype0
і
Omlp_bot_layer/mlp_bot_hiddenlayer_0/zero_fraction/cond/count_nonzero_1/NotEqualNotEqualVmlp_bot_layer/mlp_bot_hiddenlayer_0/zero_fraction/cond/count_nonzero_1/NotEqual/SwitchLmlp_bot_layer/mlp_bot_hiddenlayer_0/zero_fraction/cond/count_nonzero_1/zeros*
T0*(
_output_shapes
:€€€€€€€€€А
М
Vmlp_bot_layer/mlp_bot_hiddenlayer_0/zero_fraction/cond/count_nonzero_1/NotEqual/SwitchSwitchGmlp_bot_layer/mlp_bot_hiddenlayer_0/batch_normalization/batchnorm/add_1>mlp_bot_layer/mlp_bot_hiddenlayer_0/zero_fraction/cond/pred_id*Z
_classP
NLloc:@mlp_bot_layer/mlp_bot_hiddenlayer_0/batch_normalization/batchnorm/add_1*
T0*<
_output_shapes*
(:€€€€€€€€€А:€€€€€€€€€А
ж
Kmlp_bot_layer/mlp_bot_hiddenlayer_0/zero_fraction/cond/count_nonzero_1/CastCastOmlp_bot_layer/mlp_bot_hiddenlayer_0/zero_fraction/cond/count_nonzero_1/NotEqual*

SrcT0
*(
_output_shapes
:€€€€€€€€€А*

DstT0	
я
Lmlp_bot_layer/mlp_bot_hiddenlayer_0/zero_fraction/cond/count_nonzero_1/ConstConst@^mlp_bot_layer/mlp_bot_hiddenlayer_0/zero_fraction/cond/switch_f*
valueB"       *
dtype0*
_output_shapes
:
Ч
Tmlp_bot_layer/mlp_bot_hiddenlayer_0/zero_fraction/cond/count_nonzero_1/nonzero_countSumKmlp_bot_layer/mlp_bot_hiddenlayer_0/zero_fraction/cond/count_nonzero_1/CastLmlp_bot_layer/mlp_bot_hiddenlayer_0/zero_fraction/cond/count_nonzero_1/Const*
T0	*
_output_shapes
: 
Д
<mlp_bot_layer/mlp_bot_hiddenlayer_0/zero_fraction/cond/MergeMergeTmlp_bot_layer/mlp_bot_hiddenlayer_0/zero_fraction/cond/count_nonzero_1/nonzero_count;mlp_bot_layer/mlp_bot_hiddenlayer_0/zero_fraction/cond/Cast*
T0	*
_output_shapes
: : *
N
ж
Hmlp_bot_layer/mlp_bot_hiddenlayer_0/zero_fraction/counts_to_fraction/subSub6mlp_bot_layer/mlp_bot_hiddenlayer_0/zero_fraction/Size<mlp_bot_layer/mlp_bot_hiddenlayer_0/zero_fraction/cond/Merge*
_output_shapes
: *
T0	
Ћ
Imlp_bot_layer/mlp_bot_hiddenlayer_0/zero_fraction/counts_to_fraction/CastCastHmlp_bot_layer/mlp_bot_hiddenlayer_0/zero_fraction/counts_to_fraction/sub*

DstT0*

SrcT0	*
_output_shapes
: 
ї
Kmlp_bot_layer/mlp_bot_hiddenlayer_0/zero_fraction/counts_to_fraction/Cast_1Cast6mlp_bot_layer/mlp_bot_hiddenlayer_0/zero_fraction/Size*

DstT0*
_output_shapes
: *

SrcT0	
Р
Lmlp_bot_layer/mlp_bot_hiddenlayer_0/zero_fraction/counts_to_fraction/truedivRealDivImlp_bot_layer/mlp_bot_hiddenlayer_0/zero_fraction/counts_to_fraction/CastKmlp_bot_layer/mlp_bot_hiddenlayer_0/zero_fraction/counts_to_fraction/Cast_1*
T0*
_output_shapes
: 
µ
:mlp_bot_layer/mlp_bot_hiddenlayer_0/zero_fraction/fractionIdentityLmlp_bot_layer/mlp_bot_hiddenlayer_0/zero_fraction/counts_to_fraction/truediv*
_output_shapes
: *
T0
Д
dmlp_bot_layer/mlp_bot_hiddenlayer_0/mlp_bot_layer/mlp_bot_hiddenlayer_0/fraction_of_zero_values/tagsConst*
_output_shapes
: *
dtype0*p
valuegBe B_mlp_bot_layer/mlp_bot_hiddenlayer_0/mlp_bot_layer/mlp_bot_hiddenlayer_0/fraction_of_zero_values
≥
_mlp_bot_layer/mlp_bot_hiddenlayer_0/mlp_bot_layer/mlp_bot_hiddenlayer_0/fraction_of_zero_valuesScalarSummarydmlp_bot_layer/mlp_bot_hiddenlayer_0/mlp_bot_layer/mlp_bot_hiddenlayer_0/fraction_of_zero_values/tags:mlp_bot_layer/mlp_bot_hiddenlayer_0/zero_fraction/fraction*
_output_shapes
: *
T0
й
Vmlp_bot_layer/mlp_bot_hiddenlayer_0/mlp_bot_layer/mlp_bot_hiddenlayer_0/activation/tagConst*
_output_shapes
: *
dtype0*c
valueZBX BRmlp_bot_layer/mlp_bot_hiddenlayer_0/mlp_bot_layer/mlp_bot_hiddenlayer_0/activation
Я
Rmlp_bot_layer/mlp_bot_hiddenlayer_0/mlp_bot_layer/mlp_bot_hiddenlayer_0/activationHistogramSummaryVmlp_bot_layer/mlp_bot_hiddenlayer_0/mlp_bot_layer/mlp_bot_hiddenlayer_0/activation/tagGmlp_bot_layer/mlp_bot_hiddenlayer_0/batch_normalization/batchnorm/add_1*
_output_shapes
: 
й
Rmlp_bot_layer/mlp_bot_hiddenlayer_1/kernel/part_0/Initializer/random_uniform/shapeConst*
valueB"      *
dtype0*D
_class:
86loc:@mlp_bot_layer/mlp_bot_hiddenlayer_1/kernel/part_0*
_output_shapes
:
џ
Pmlp_bot_layer/mlp_bot_hiddenlayer_1/kernel/part_0/Initializer/random_uniform/minConst*
valueB
 *уµљ*
_output_shapes
: *
dtype0*D
_class:
86loc:@mlp_bot_layer/mlp_bot_hiddenlayer_1/kernel/part_0
џ
Pmlp_bot_layer/mlp_bot_hiddenlayer_1/kernel/part_0/Initializer/random_uniform/maxConst*
valueB
 *уµ=*
_output_shapes
: *D
_class:
86loc:@mlp_bot_layer/mlp_bot_hiddenlayer_1/kernel/part_0*
dtype0
љ
Zmlp_bot_layer/mlp_bot_hiddenlayer_1/kernel/part_0/Initializer/random_uniform/RandomUniformRandomUniformRmlp_bot_layer/mlp_bot_hiddenlayer_1/kernel/part_0/Initializer/random_uniform/shape* 
_output_shapes
:
АА*
dtype0*
T0*D
_class:
86loc:@mlp_bot_layer/mlp_bot_hiddenlayer_1/kernel/part_0
в
Pmlp_bot_layer/mlp_bot_hiddenlayer_1/kernel/part_0/Initializer/random_uniform/subSubPmlp_bot_layer/mlp_bot_hiddenlayer_1/kernel/part_0/Initializer/random_uniform/maxPmlp_bot_layer/mlp_bot_hiddenlayer_1/kernel/part_0/Initializer/random_uniform/min*D
_class:
86loc:@mlp_bot_layer/mlp_bot_hiddenlayer_1/kernel/part_0*
T0*
_output_shapes
: 
ц
Pmlp_bot_layer/mlp_bot_hiddenlayer_1/kernel/part_0/Initializer/random_uniform/mulMulZmlp_bot_layer/mlp_bot_hiddenlayer_1/kernel/part_0/Initializer/random_uniform/RandomUniformPmlp_bot_layer/mlp_bot_hiddenlayer_1/kernel/part_0/Initializer/random_uniform/sub*D
_class:
86loc:@mlp_bot_layer/mlp_bot_hiddenlayer_1/kernel/part_0*
T0* 
_output_shapes
:
АА
и
Lmlp_bot_layer/mlp_bot_hiddenlayer_1/kernel/part_0/Initializer/random_uniformAddPmlp_bot_layer/mlp_bot_hiddenlayer_1/kernel/part_0/Initializer/random_uniform/mulPmlp_bot_layer/mlp_bot_hiddenlayer_1/kernel/part_0/Initializer/random_uniform/min*
T0* 
_output_shapes
:
АА*D
_class:
86loc:@mlp_bot_layer/mlp_bot_hiddenlayer_1/kernel/part_0
Ћ
1mlp_bot_layer/mlp_bot_hiddenlayer_1/kernel/part_0
VariableV2*
shape:
АА*
dtype0*D
_class:
86loc:@mlp_bot_layer/mlp_bot_hiddenlayer_1/kernel/part_0* 
_output_shapes
:
АА
і
8mlp_bot_layer/mlp_bot_hiddenlayer_1/kernel/part_0/AssignAssign1mlp_bot_layer/mlp_bot_hiddenlayer_1/kernel/part_0Lmlp_bot_layer/mlp_bot_hiddenlayer_1/kernel/part_0/Initializer/random_uniform*
T0* 
_output_shapes
:
АА*D
_class:
86loc:@mlp_bot_layer/mlp_bot_hiddenlayer_1/kernel/part_0
ж
6mlp_bot_layer/mlp_bot_hiddenlayer_1/kernel/part_0/readIdentity1mlp_bot_layer/mlp_bot_hiddenlayer_1/kernel/part_0* 
_output_shapes
:
АА*
T0*D
_class:
86loc:@mlp_bot_layer/mlp_bot_hiddenlayer_1/kernel/part_0
‘
Amlp_bot_layer/mlp_bot_hiddenlayer_1/bias/part_0/Initializer/zerosConst*B
_class8
64loc:@mlp_bot_layer/mlp_bot_hiddenlayer_1/bias/part_0*
_output_shapes	
:А*
dtype0*
valueBА*    
љ
/mlp_bot_layer/mlp_bot_hiddenlayer_1/bias/part_0
VariableV2*
_output_shapes	
:А*
shape:А*B
_class8
64loc:@mlp_bot_layer/mlp_bot_hiddenlayer_1/bias/part_0*
dtype0
Ю
6mlp_bot_layer/mlp_bot_hiddenlayer_1/bias/part_0/AssignAssign/mlp_bot_layer/mlp_bot_hiddenlayer_1/bias/part_0Amlp_bot_layer/mlp_bot_hiddenlayer_1/bias/part_0/Initializer/zeros*B
_class8
64loc:@mlp_bot_layer/mlp_bot_hiddenlayer_1/bias/part_0*
T0*
_output_shapes	
:А
џ
4mlp_bot_layer/mlp_bot_hiddenlayer_1/bias/part_0/readIdentity/mlp_bot_layer/mlp_bot_hiddenlayer_1/bias/part_0*B
_class8
64loc:@mlp_bot_layer/mlp_bot_hiddenlayer_1/bias/part_0*
_output_shapes	
:А*
T0
Щ
*mlp_bot_layer/mlp_bot_hiddenlayer_1/kernelIdentity6mlp_bot_layer/mlp_bot_hiddenlayer_1/kernel/part_0/read* 
_output_shapes
:
АА*
T0
№
*mlp_bot_layer/mlp_bot_hiddenlayer_1/MatMulMatMulGmlp_bot_layer/mlp_bot_hiddenlayer_0/batch_normalization/batchnorm/add_1*mlp_bot_layer/mlp_bot_hiddenlayer_1/kernel*(
_output_shapes
:€€€€€€€€€А*
T0
Р
(mlp_bot_layer/mlp_bot_hiddenlayer_1/biasIdentity4mlp_bot_layer/mlp_bot_hiddenlayer_1/bias/part_0/read*
_output_shapes	
:А*
T0
њ
+mlp_bot_layer/mlp_bot_hiddenlayer_1/BiasAddBiasAdd*mlp_bot_layer/mlp_bot_hiddenlayer_1/MatMul(mlp_bot_layer/mlp_bot_hiddenlayer_1/bias*(
_output_shapes
:€€€€€€€€€А*
T0
Р
(mlp_bot_layer/mlp_bot_hiddenlayer_1/ReluRelu+mlp_bot_layer/mlp_bot_hiddenlayer_1/BiasAdd*
T0*(
_output_shapes
:€€€€€€€€€А
э
Umlp_bot_layer/mlp_bot_hiddenlayer_1/batch_normalization/gamma/part_0/Initializer/onesConst*
dtype0*
valueBА*  А?*W
_classM
KIloc:@mlp_bot_layer/mlp_bot_hiddenlayer_1/batch_normalization/gamma/part_0*
_output_shapes	
:А
з
Dmlp_bot_layer/mlp_bot_hiddenlayer_1/batch_normalization/gamma/part_0
VariableV2*
_output_shapes	
:А*W
_classM
KIloc:@mlp_bot_layer/mlp_bot_hiddenlayer_1/batch_normalization/gamma/part_0*
dtype0*
shape:А
с
Kmlp_bot_layer/mlp_bot_hiddenlayer_1/batch_normalization/gamma/part_0/AssignAssignDmlp_bot_layer/mlp_bot_hiddenlayer_1/batch_normalization/gamma/part_0Umlp_bot_layer/mlp_bot_hiddenlayer_1/batch_normalization/gamma/part_0/Initializer/ones*
T0*
_output_shapes	
:А*W
_classM
KIloc:@mlp_bot_layer/mlp_bot_hiddenlayer_1/batch_normalization/gamma/part_0
Ъ
Imlp_bot_layer/mlp_bot_hiddenlayer_1/batch_normalization/gamma/part_0/readIdentityDmlp_bot_layer/mlp_bot_hiddenlayer_1/batch_normalization/gamma/part_0*
_output_shapes	
:А*W
_classM
KIloc:@mlp_bot_layer/mlp_bot_hiddenlayer_1/batch_normalization/gamma/part_0*
T0
ь
Umlp_bot_layer/mlp_bot_hiddenlayer_1/batch_normalization/beta/part_0/Initializer/zerosConst*
dtype0*V
_classL
JHloc:@mlp_bot_layer/mlp_bot_hiddenlayer_1/batch_normalization/beta/part_0*
valueBА*    *
_output_shapes	
:А
е
Cmlp_bot_layer/mlp_bot_hiddenlayer_1/batch_normalization/beta/part_0
VariableV2*
dtype0*
_output_shapes	
:А*V
_classL
JHloc:@mlp_bot_layer/mlp_bot_hiddenlayer_1/batch_normalization/beta/part_0*
shape:А
о
Jmlp_bot_layer/mlp_bot_hiddenlayer_1/batch_normalization/beta/part_0/AssignAssignCmlp_bot_layer/mlp_bot_hiddenlayer_1/batch_normalization/beta/part_0Umlp_bot_layer/mlp_bot_hiddenlayer_1/batch_normalization/beta/part_0/Initializer/zeros*
_output_shapes	
:А*V
_classL
JHloc:@mlp_bot_layer/mlp_bot_hiddenlayer_1/batch_normalization/beta/part_0*
T0
Ч
Hmlp_bot_layer/mlp_bot_hiddenlayer_1/batch_normalization/beta/part_0/readIdentityCmlp_bot_layer/mlp_bot_hiddenlayer_1/batch_normalization/beta/part_0*
T0*
_output_shapes	
:А*V
_classL
JHloc:@mlp_bot_layer/mlp_bot_hiddenlayer_1/batch_normalization/beta/part_0
ь
Umlp_bot_layer/mlp_bot_hiddenlayer_1/batch_normalization/moving_mean/Initializer/zerosConst*
dtype0*
_output_shapes	
:А*
valueBА*    *V
_classL
JHloc:@mlp_bot_layer/mlp_bot_hiddenlayer_1/batch_normalization/moving_mean
е
Cmlp_bot_layer/mlp_bot_hiddenlayer_1/batch_normalization/moving_mean
VariableV2*V
_classL
JHloc:@mlp_bot_layer/mlp_bot_hiddenlayer_1/batch_normalization/moving_mean*
dtype0*
shape:А*
_output_shapes	
:А
о
Jmlp_bot_layer/mlp_bot_hiddenlayer_1/batch_normalization/moving_mean/AssignAssignCmlp_bot_layer/mlp_bot_hiddenlayer_1/batch_normalization/moving_meanUmlp_bot_layer/mlp_bot_hiddenlayer_1/batch_normalization/moving_mean/Initializer/zeros*
_output_shapes	
:А*V
_classL
JHloc:@mlp_bot_layer/mlp_bot_hiddenlayer_1/batch_normalization/moving_mean*
T0
Ч
Hmlp_bot_layer/mlp_bot_hiddenlayer_1/batch_normalization/moving_mean/readIdentityCmlp_bot_layer/mlp_bot_hiddenlayer_1/batch_normalization/moving_mean*
_output_shapes	
:А*V
_classL
JHloc:@mlp_bot_layer/mlp_bot_hiddenlayer_1/batch_normalization/moving_mean*
T0
Г
Xmlp_bot_layer/mlp_bot_hiddenlayer_1/batch_normalization/moving_variance/Initializer/onesConst*Z
_classP
NLloc:@mlp_bot_layer/mlp_bot_hiddenlayer_1/batch_normalization/moving_variance*
valueBА*  А?*
_output_shapes	
:А*
dtype0
н
Gmlp_bot_layer/mlp_bot_hiddenlayer_1/batch_normalization/moving_variance
VariableV2*
shape:А*
dtype0*
_output_shapes	
:А*Z
_classP
NLloc:@mlp_bot_layer/mlp_bot_hiddenlayer_1/batch_normalization/moving_variance
э
Nmlp_bot_layer/mlp_bot_hiddenlayer_1/batch_normalization/moving_variance/AssignAssignGmlp_bot_layer/mlp_bot_hiddenlayer_1/batch_normalization/moving_varianceXmlp_bot_layer/mlp_bot_hiddenlayer_1/batch_normalization/moving_variance/Initializer/ones*Z
_classP
NLloc:@mlp_bot_layer/mlp_bot_hiddenlayer_1/batch_normalization/moving_variance*
T0*
_output_shapes	
:А
£
Lmlp_bot_layer/mlp_bot_hiddenlayer_1/batch_normalization/moving_variance/readIdentityGmlp_bot_layer/mlp_bot_hiddenlayer_1/batch_normalization/moving_variance*
T0*
_output_shapes	
:А*Z
_classP
NLloc:@mlp_bot_layer/mlp_bot_hiddenlayer_1/batch_normalization/moving_variance
†
Vmlp_bot_layer/mlp_bot_hiddenlayer_1/batch_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
valueB: *
dtype0
Й
Dmlp_bot_layer/mlp_bot_hiddenlayer_1/batch_normalization/moments/meanMean(mlp_bot_layer/mlp_bot_hiddenlayer_1/ReluVmlp_bot_layer/mlp_bot_hiddenlayer_1/batch_normalization/moments/mean/reduction_indices*
_output_shapes
:	А*
T0*
	keep_dims(
ћ
Lmlp_bot_layer/mlp_bot_hiddenlayer_1/batch_normalization/moments/StopGradientStopGradientDmlp_bot_layer/mlp_bot_hiddenlayer_1/batch_normalization/moments/mean*
_output_shapes
:	А*
T0
С
Qmlp_bot_layer/mlp_bot_hiddenlayer_1/batch_normalization/moments/SquaredDifferenceSquaredDifference(mlp_bot_layer/mlp_bot_hiddenlayer_1/ReluLmlp_bot_layer/mlp_bot_hiddenlayer_1/batch_normalization/moments/StopGradient*
T0*(
_output_shapes
:€€€€€€€€€А
§
Zmlp_bot_layer/mlp_bot_hiddenlayer_1/batch_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
Ї
Hmlp_bot_layer/mlp_bot_hiddenlayer_1/batch_normalization/moments/varianceMeanQmlp_bot_layer/mlp_bot_hiddenlayer_1/batch_normalization/moments/SquaredDifferenceZmlp_bot_layer/mlp_bot_hiddenlayer_1/batch_normalization/moments/variance/reduction_indices*
	keep_dims(*
T0*
_output_shapes
:	А
’
Gmlp_bot_layer/mlp_bot_hiddenlayer_1/batch_normalization/moments/SqueezeSqueezeDmlp_bot_layer/mlp_bot_hiddenlayer_1/batch_normalization/moments/mean*
_output_shapes	
:А*
T0*
squeeze_dims
 
џ
Imlp_bot_layer/mlp_bot_hiddenlayer_1/batch_normalization/moments/Squeeze_1SqueezeHmlp_bot_layer/mlp_bot_hiddenlayer_1/batch_normalization/moments/variance*
T0*
squeeze_dims
 *
_output_shapes	
:А
к
Mmlp_bot_layer/mlp_bot_hiddenlayer_1/batch_normalization/AssignMovingAvg/decayConst*
_output_shapes
: *V
_classL
JHloc:@mlp_bot_layer/mlp_bot_hiddenlayer_1/batch_normalization/moving_mean*
dtype0*
valueB
 *
„#<
г
Kmlp_bot_layer/mlp_bot_hiddenlayer_1/batch_normalization/AssignMovingAvg/subSubHmlp_bot_layer/mlp_bot_hiddenlayer_1/batch_normalization/moving_mean/readGmlp_bot_layer/mlp_bot_hiddenlayer_1/batch_normalization/moments/Squeeze*
_output_shapes	
:А*
T0*V
_classL
JHloc:@mlp_bot_layer/mlp_bot_hiddenlayer_1/batch_normalization/moving_mean
м
Kmlp_bot_layer/mlp_bot_hiddenlayer_1/batch_normalization/AssignMovingAvg/mulMulKmlp_bot_layer/mlp_bot_hiddenlayer_1/batch_normalization/AssignMovingAvg/subMmlp_bot_layer/mlp_bot_hiddenlayer_1/batch_normalization/AssignMovingAvg/decay*
_output_shapes	
:А*
T0*V
_classL
JHloc:@mlp_bot_layer/mlp_bot_hiddenlayer_1/batch_normalization/moving_mean
д
Gmlp_bot_layer/mlp_bot_hiddenlayer_1/batch_normalization/AssignMovingAvg	AssignSubCmlp_bot_layer/mlp_bot_hiddenlayer_1/batch_normalization/moving_meanKmlp_bot_layer/mlp_bot_hiddenlayer_1/batch_normalization/AssignMovingAvg/mul*
T0*
_output_shapes	
:А*V
_classL
JHloc:@mlp_bot_layer/mlp_bot_hiddenlayer_1/batch_normalization/moving_mean
р
Omlp_bot_layer/mlp_bot_hiddenlayer_1/batch_normalization/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*Z
_classP
NLloc:@mlp_bot_layer/mlp_bot_hiddenlayer_1/batch_normalization/moving_variance*
valueB
 *
„#<
п
Mmlp_bot_layer/mlp_bot_hiddenlayer_1/batch_normalization/AssignMovingAvg_1/subSubLmlp_bot_layer/mlp_bot_hiddenlayer_1/batch_normalization/moving_variance/readImlp_bot_layer/mlp_bot_hiddenlayer_1/batch_normalization/moments/Squeeze_1*
_output_shapes	
:А*Z
_classP
NLloc:@mlp_bot_layer/mlp_bot_hiddenlayer_1/batch_normalization/moving_variance*
T0
ц
Mmlp_bot_layer/mlp_bot_hiddenlayer_1/batch_normalization/AssignMovingAvg_1/mulMulMmlp_bot_layer/mlp_bot_hiddenlayer_1/batch_normalization/AssignMovingAvg_1/subOmlp_bot_layer/mlp_bot_hiddenlayer_1/batch_normalization/AssignMovingAvg_1/decay*
T0*Z
_classP
NLloc:@mlp_bot_layer/mlp_bot_hiddenlayer_1/batch_normalization/moving_variance*
_output_shapes	
:А
р
Imlp_bot_layer/mlp_bot_hiddenlayer_1/batch_normalization/AssignMovingAvg_1	AssignSubGmlp_bot_layer/mlp_bot_hiddenlayer_1/batch_normalization/moving_varianceMmlp_bot_layer/mlp_bot_hiddenlayer_1/batch_normalization/AssignMovingAvg_1/mul*
_output_shapes	
:А*Z
_classP
NLloc:@mlp_bot_layer/mlp_bot_hiddenlayer_1/batch_normalization/moving_variance*
T0
Є
<mlp_bot_layer/mlp_bot_hiddenlayer_1/batch_normalization/betaIdentityHmlp_bot_layer/mlp_bot_hiddenlayer_1/batch_normalization/beta/part_0/read*
T0*
_output_shapes	
:А
Ї
=mlp_bot_layer/mlp_bot_hiddenlayer_1/batch_normalization/gammaIdentityImlp_bot_layer/mlp_bot_hiddenlayer_1/batch_normalization/gamma/part_0/read*
T0*
_output_shapes	
:А
М
Gmlp_bot_layer/mlp_bot_hiddenlayer_1/batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:
И
Emlp_bot_layer/mlp_bot_hiddenlayer_1/batch_normalization/batchnorm/addAddV2Imlp_bot_layer/mlp_bot_hiddenlayer_1/batch_normalization/moments/Squeeze_1Gmlp_bot_layer/mlp_bot_hiddenlayer_1/batch_normalization/batchnorm/add/y*
_output_shapes	
:А*
T0
љ
Gmlp_bot_layer/mlp_bot_hiddenlayer_1/batch_normalization/batchnorm/RsqrtRsqrtEmlp_bot_layer/mlp_bot_hiddenlayer_1/batch_normalization/batchnorm/add*
T0*
_output_shapes	
:А
ъ
Emlp_bot_layer/mlp_bot_hiddenlayer_1/batch_normalization/batchnorm/mulMulGmlp_bot_layer/mlp_bot_hiddenlayer_1/batch_normalization/batchnorm/Rsqrt=mlp_bot_layer/mlp_bot_hiddenlayer_1/batch_normalization/gamma*
T0*
_output_shapes	
:А
т
Gmlp_bot_layer/mlp_bot_hiddenlayer_1/batch_normalization/batchnorm/mul_1Mul(mlp_bot_layer/mlp_bot_hiddenlayer_1/ReluEmlp_bot_layer/mlp_bot_hiddenlayer_1/batch_normalization/batchnorm/mul*
T0*(
_output_shapes
:€€€€€€€€€А
Д
Gmlp_bot_layer/mlp_bot_hiddenlayer_1/batch_normalization/batchnorm/mul_2MulGmlp_bot_layer/mlp_bot_hiddenlayer_1/batch_normalization/moments/SqueezeEmlp_bot_layer/mlp_bot_hiddenlayer_1/batch_normalization/batchnorm/mul*
_output_shapes	
:А*
T0
щ
Emlp_bot_layer/mlp_bot_hiddenlayer_1/batch_normalization/batchnorm/subSub<mlp_bot_layer/mlp_bot_hiddenlayer_1/batch_normalization/betaGmlp_bot_layer/mlp_bot_hiddenlayer_1/batch_normalization/batchnorm/mul_2*
T0*
_output_shapes	
:А
У
Gmlp_bot_layer/mlp_bot_hiddenlayer_1/batch_normalization/batchnorm/add_1AddV2Gmlp_bot_layer/mlp_bot_hiddenlayer_1/batch_normalization/batchnorm/mul_1Emlp_bot_layer/mlp_bot_hiddenlayer_1/batch_normalization/batchnorm/sub*(
_output_shapes
:€€€€€€€€€А*
T0
Є
6mlp_bot_layer/mlp_bot_hiddenlayer_1/zero_fraction/SizeSizeGmlp_bot_layer/mlp_bot_hiddenlayer_1/batch_normalization/batchnorm/add_1*
_output_shapes
: *
T0*
out_type0	
Г
=mlp_bot_layer/mlp_bot_hiddenlayer_1/zero_fraction/LessEqual/yConst*
valueB	 R€€€€*
_output_shapes
: *
dtype0	
а
;mlp_bot_layer/mlp_bot_hiddenlayer_1/zero_fraction/LessEqual	LessEqual6mlp_bot_layer/mlp_bot_hiddenlayer_1/zero_fraction/Size=mlp_bot_layer/mlp_bot_hiddenlayer_1/zero_fraction/LessEqual/y*
T0	*
_output_shapes
: 
д
=mlp_bot_layer/mlp_bot_hiddenlayer_1/zero_fraction/cond/SwitchSwitch;mlp_bot_layer/mlp_bot_hiddenlayer_1/zero_fraction/LessEqual;mlp_bot_layer/mlp_bot_hiddenlayer_1/zero_fraction/LessEqual*
T0
*
_output_shapes
: : 
≠
?mlp_bot_layer/mlp_bot_hiddenlayer_1/zero_fraction/cond/switch_tIdentity?mlp_bot_layer/mlp_bot_hiddenlayer_1/zero_fraction/cond/Switch:1*
T0
*
_output_shapes
: 
Ђ
?mlp_bot_layer/mlp_bot_hiddenlayer_1/zero_fraction/cond/switch_fIdentity=mlp_bot_layer/mlp_bot_hiddenlayer_1/zero_fraction/cond/Switch*
_output_shapes
: *
T0

®
>mlp_bot_layer/mlp_bot_hiddenlayer_1/zero_fraction/cond/pred_idIdentity;mlp_bot_layer/mlp_bot_hiddenlayer_1/zero_fraction/LessEqual*
T0
*
_output_shapes
: 
—
Jmlp_bot_layer/mlp_bot_hiddenlayer_1/zero_fraction/cond/count_nonzero/zerosConst@^mlp_bot_layer/mlp_bot_hiddenlayer_1/zero_fraction/cond/switch_t*
valueB
 *    *
dtype0*
_output_shapes
: 
∞
Mmlp_bot_layer/mlp_bot_hiddenlayer_1/zero_fraction/cond/count_nonzero/NotEqualNotEqualVmlp_bot_layer/mlp_bot_hiddenlayer_1/zero_fraction/cond/count_nonzero/NotEqual/Switch:1Jmlp_bot_layer/mlp_bot_hiddenlayer_1/zero_fraction/cond/count_nonzero/zeros*(
_output_shapes
:€€€€€€€€€А*
T0
К
Tmlp_bot_layer/mlp_bot_hiddenlayer_1/zero_fraction/cond/count_nonzero/NotEqual/SwitchSwitchGmlp_bot_layer/mlp_bot_hiddenlayer_1/batch_normalization/batchnorm/add_1>mlp_bot_layer/mlp_bot_hiddenlayer_1/zero_fraction/cond/pred_id*<
_output_shapes*
(:€€€€€€€€€А:€€€€€€€€€А*Z
_classP
NLloc:@mlp_bot_layer/mlp_bot_hiddenlayer_1/batch_normalization/batchnorm/add_1*
T0
в
Imlp_bot_layer/mlp_bot_hiddenlayer_1/zero_fraction/cond/count_nonzero/CastCastMmlp_bot_layer/mlp_bot_hiddenlayer_1/zero_fraction/cond/count_nonzero/NotEqual*

DstT0*(
_output_shapes
:€€€€€€€€€А*

SrcT0

Ё
Jmlp_bot_layer/mlp_bot_hiddenlayer_1/zero_fraction/cond/count_nonzero/ConstConst@^mlp_bot_layer/mlp_bot_hiddenlayer_1/zero_fraction/cond/switch_t*
valueB"       *
dtype0*
_output_shapes
:
С
Rmlp_bot_layer/mlp_bot_hiddenlayer_1/zero_fraction/cond/count_nonzero/nonzero_countSumImlp_bot_layer/mlp_bot_hiddenlayer_1/zero_fraction/cond/count_nonzero/CastJmlp_bot_layer/mlp_bot_hiddenlayer_1/zero_fraction/cond/count_nonzero/Const*
_output_shapes
: *
T0
«
;mlp_bot_layer/mlp_bot_hiddenlayer_1/zero_fraction/cond/CastCastRmlp_bot_layer/mlp_bot_hiddenlayer_1/zero_fraction/cond/count_nonzero/nonzero_count*

SrcT0*
_output_shapes
: *

DstT0	
”
Lmlp_bot_layer/mlp_bot_hiddenlayer_1/zero_fraction/cond/count_nonzero_1/zerosConst@^mlp_bot_layer/mlp_bot_hiddenlayer_1/zero_fraction/cond/switch_f*
_output_shapes
: *
valueB
 *    *
dtype0
і
Omlp_bot_layer/mlp_bot_hiddenlayer_1/zero_fraction/cond/count_nonzero_1/NotEqualNotEqualVmlp_bot_layer/mlp_bot_hiddenlayer_1/zero_fraction/cond/count_nonzero_1/NotEqual/SwitchLmlp_bot_layer/mlp_bot_hiddenlayer_1/zero_fraction/cond/count_nonzero_1/zeros*(
_output_shapes
:€€€€€€€€€А*
T0
М
Vmlp_bot_layer/mlp_bot_hiddenlayer_1/zero_fraction/cond/count_nonzero_1/NotEqual/SwitchSwitchGmlp_bot_layer/mlp_bot_hiddenlayer_1/batch_normalization/batchnorm/add_1>mlp_bot_layer/mlp_bot_hiddenlayer_1/zero_fraction/cond/pred_id*<
_output_shapes*
(:€€€€€€€€€А:€€€€€€€€€А*
T0*Z
_classP
NLloc:@mlp_bot_layer/mlp_bot_hiddenlayer_1/batch_normalization/batchnorm/add_1
ж
Kmlp_bot_layer/mlp_bot_hiddenlayer_1/zero_fraction/cond/count_nonzero_1/CastCastOmlp_bot_layer/mlp_bot_hiddenlayer_1/zero_fraction/cond/count_nonzero_1/NotEqual*(
_output_shapes
:€€€€€€€€€А*

SrcT0
*

DstT0	
я
Lmlp_bot_layer/mlp_bot_hiddenlayer_1/zero_fraction/cond/count_nonzero_1/ConstConst@^mlp_bot_layer/mlp_bot_hiddenlayer_1/zero_fraction/cond/switch_f*
valueB"       *
_output_shapes
:*
dtype0
Ч
Tmlp_bot_layer/mlp_bot_hiddenlayer_1/zero_fraction/cond/count_nonzero_1/nonzero_countSumKmlp_bot_layer/mlp_bot_hiddenlayer_1/zero_fraction/cond/count_nonzero_1/CastLmlp_bot_layer/mlp_bot_hiddenlayer_1/zero_fraction/cond/count_nonzero_1/Const*
T0	*
_output_shapes
: 
Д
<mlp_bot_layer/mlp_bot_hiddenlayer_1/zero_fraction/cond/MergeMergeTmlp_bot_layer/mlp_bot_hiddenlayer_1/zero_fraction/cond/count_nonzero_1/nonzero_count;mlp_bot_layer/mlp_bot_hiddenlayer_1/zero_fraction/cond/Cast*
_output_shapes
: : *
N*
T0	
ж
Hmlp_bot_layer/mlp_bot_hiddenlayer_1/zero_fraction/counts_to_fraction/subSub6mlp_bot_layer/mlp_bot_hiddenlayer_1/zero_fraction/Size<mlp_bot_layer/mlp_bot_hiddenlayer_1/zero_fraction/cond/Merge*
_output_shapes
: *
T0	
Ћ
Imlp_bot_layer/mlp_bot_hiddenlayer_1/zero_fraction/counts_to_fraction/CastCastHmlp_bot_layer/mlp_bot_hiddenlayer_1/zero_fraction/counts_to_fraction/sub*
_output_shapes
: *

DstT0*

SrcT0	
ї
Kmlp_bot_layer/mlp_bot_hiddenlayer_1/zero_fraction/counts_to_fraction/Cast_1Cast6mlp_bot_layer/mlp_bot_hiddenlayer_1/zero_fraction/Size*
_output_shapes
: *

SrcT0	*

DstT0
Р
Lmlp_bot_layer/mlp_bot_hiddenlayer_1/zero_fraction/counts_to_fraction/truedivRealDivImlp_bot_layer/mlp_bot_hiddenlayer_1/zero_fraction/counts_to_fraction/CastKmlp_bot_layer/mlp_bot_hiddenlayer_1/zero_fraction/counts_to_fraction/Cast_1*
T0*
_output_shapes
: 
µ
:mlp_bot_layer/mlp_bot_hiddenlayer_1/zero_fraction/fractionIdentityLmlp_bot_layer/mlp_bot_hiddenlayer_1/zero_fraction/counts_to_fraction/truediv*
T0*
_output_shapes
: 
Д
dmlp_bot_layer/mlp_bot_hiddenlayer_1/mlp_bot_layer/mlp_bot_hiddenlayer_1/fraction_of_zero_values/tagsConst*p
valuegBe B_mlp_bot_layer/mlp_bot_hiddenlayer_1/mlp_bot_layer/mlp_bot_hiddenlayer_1/fraction_of_zero_values*
dtype0*
_output_shapes
: 
≥
_mlp_bot_layer/mlp_bot_hiddenlayer_1/mlp_bot_layer/mlp_bot_hiddenlayer_1/fraction_of_zero_valuesScalarSummarydmlp_bot_layer/mlp_bot_hiddenlayer_1/mlp_bot_layer/mlp_bot_hiddenlayer_1/fraction_of_zero_values/tags:mlp_bot_layer/mlp_bot_hiddenlayer_1/zero_fraction/fraction*
T0*
_output_shapes
: 
й
Vmlp_bot_layer/mlp_bot_hiddenlayer_1/mlp_bot_layer/mlp_bot_hiddenlayer_1/activation/tagConst*c
valueZBX BRmlp_bot_layer/mlp_bot_hiddenlayer_1/mlp_bot_layer/mlp_bot_hiddenlayer_1/activation*
dtype0*
_output_shapes
: 
Я
Rmlp_bot_layer/mlp_bot_hiddenlayer_1/mlp_bot_layer/mlp_bot_hiddenlayer_1/activationHistogramSummaryVmlp_bot_layer/mlp_bot_hiddenlayer_1/mlp_bot_layer/mlp_bot_hiddenlayer_1/activation/tagGmlp_bot_layer/mlp_bot_hiddenlayer_1/batch_normalization/batchnorm/add_1*
_output_shapes
: 
й
Rmlp_bot_layer/mlp_bot_hiddenlayer_2/kernel/part_0/Initializer/random_uniform/shapeConst*
valueB"   @   *D
_class:
86loc:@mlp_bot_layer/mlp_bot_hiddenlayer_2/kernel/part_0*
dtype0*
_output_shapes
:
џ
Pmlp_bot_layer/mlp_bot_hiddenlayer_2/kernel/part_0/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *D
_class:
86loc:@mlp_bot_layer/mlp_bot_hiddenlayer_2/kernel/part_0*
valueB
 *М7Њ
џ
Pmlp_bot_layer/mlp_bot_hiddenlayer_2/kernel/part_0/Initializer/random_uniform/maxConst*
dtype0*
valueB
 *М7>*D
_class:
86loc:@mlp_bot_layer/mlp_bot_hiddenlayer_2/kernel/part_0*
_output_shapes
: 
Љ
Zmlp_bot_layer/mlp_bot_hiddenlayer_2/kernel/part_0/Initializer/random_uniform/RandomUniformRandomUniformRmlp_bot_layer/mlp_bot_hiddenlayer_2/kernel/part_0/Initializer/random_uniform/shape*D
_class:
86loc:@mlp_bot_layer/mlp_bot_hiddenlayer_2/kernel/part_0*
T0*
dtype0*
_output_shapes
:	А@
в
Pmlp_bot_layer/mlp_bot_hiddenlayer_2/kernel/part_0/Initializer/random_uniform/subSubPmlp_bot_layer/mlp_bot_hiddenlayer_2/kernel/part_0/Initializer/random_uniform/maxPmlp_bot_layer/mlp_bot_hiddenlayer_2/kernel/part_0/Initializer/random_uniform/min*D
_class:
86loc:@mlp_bot_layer/mlp_bot_hiddenlayer_2/kernel/part_0*
T0*
_output_shapes
: 
х
Pmlp_bot_layer/mlp_bot_hiddenlayer_2/kernel/part_0/Initializer/random_uniform/mulMulZmlp_bot_layer/mlp_bot_hiddenlayer_2/kernel/part_0/Initializer/random_uniform/RandomUniformPmlp_bot_layer/mlp_bot_hiddenlayer_2/kernel/part_0/Initializer/random_uniform/sub*D
_class:
86loc:@mlp_bot_layer/mlp_bot_hiddenlayer_2/kernel/part_0*
_output_shapes
:	А@*
T0
з
Lmlp_bot_layer/mlp_bot_hiddenlayer_2/kernel/part_0/Initializer/random_uniformAddPmlp_bot_layer/mlp_bot_hiddenlayer_2/kernel/part_0/Initializer/random_uniform/mulPmlp_bot_layer/mlp_bot_hiddenlayer_2/kernel/part_0/Initializer/random_uniform/min*
_output_shapes
:	А@*D
_class:
86loc:@mlp_bot_layer/mlp_bot_hiddenlayer_2/kernel/part_0*
T0
…
1mlp_bot_layer/mlp_bot_hiddenlayer_2/kernel/part_0
VariableV2*
_output_shapes
:	А@*
dtype0*
shape:	А@*D
_class:
86loc:@mlp_bot_layer/mlp_bot_hiddenlayer_2/kernel/part_0
≥
8mlp_bot_layer/mlp_bot_hiddenlayer_2/kernel/part_0/AssignAssign1mlp_bot_layer/mlp_bot_hiddenlayer_2/kernel/part_0Lmlp_bot_layer/mlp_bot_hiddenlayer_2/kernel/part_0/Initializer/random_uniform*
_output_shapes
:	А@*
T0*D
_class:
86loc:@mlp_bot_layer/mlp_bot_hiddenlayer_2/kernel/part_0
е
6mlp_bot_layer/mlp_bot_hiddenlayer_2/kernel/part_0/readIdentity1mlp_bot_layer/mlp_bot_hiddenlayer_2/kernel/part_0*D
_class:
86loc:@mlp_bot_layer/mlp_bot_hiddenlayer_2/kernel/part_0*
T0*
_output_shapes
:	А@
“
Amlp_bot_layer/mlp_bot_hiddenlayer_2/bias/part_0/Initializer/zerosConst*
valueB@*    *
_output_shapes
:@*B
_class8
64loc:@mlp_bot_layer/mlp_bot_hiddenlayer_2/bias/part_0*
dtype0
ї
/mlp_bot_layer/mlp_bot_hiddenlayer_2/bias/part_0
VariableV2*
_output_shapes
:@*
dtype0*B
_class8
64loc:@mlp_bot_layer/mlp_bot_hiddenlayer_2/bias/part_0*
shape:@
Э
6mlp_bot_layer/mlp_bot_hiddenlayer_2/bias/part_0/AssignAssign/mlp_bot_layer/mlp_bot_hiddenlayer_2/bias/part_0Amlp_bot_layer/mlp_bot_hiddenlayer_2/bias/part_0/Initializer/zeros*
_output_shapes
:@*
T0*B
_class8
64loc:@mlp_bot_layer/mlp_bot_hiddenlayer_2/bias/part_0
Џ
4mlp_bot_layer/mlp_bot_hiddenlayer_2/bias/part_0/readIdentity/mlp_bot_layer/mlp_bot_hiddenlayer_2/bias/part_0*
_output_shapes
:@*
T0*B
_class8
64loc:@mlp_bot_layer/mlp_bot_hiddenlayer_2/bias/part_0
Ш
*mlp_bot_layer/mlp_bot_hiddenlayer_2/kernelIdentity6mlp_bot_layer/mlp_bot_hiddenlayer_2/kernel/part_0/read*
_output_shapes
:	А@*
T0
џ
*mlp_bot_layer/mlp_bot_hiddenlayer_2/MatMulMatMulGmlp_bot_layer/mlp_bot_hiddenlayer_1/batch_normalization/batchnorm/add_1*mlp_bot_layer/mlp_bot_hiddenlayer_2/kernel*'
_output_shapes
:€€€€€€€€€@*
T0
П
(mlp_bot_layer/mlp_bot_hiddenlayer_2/biasIdentity4mlp_bot_layer/mlp_bot_hiddenlayer_2/bias/part_0/read*
T0*
_output_shapes
:@
Њ
+mlp_bot_layer/mlp_bot_hiddenlayer_2/BiasAddBiasAdd*mlp_bot_layer/mlp_bot_hiddenlayer_2/MatMul(mlp_bot_layer/mlp_bot_hiddenlayer_2/bias*
T0*'
_output_shapes
:€€€€€€€€€@
П
(mlp_bot_layer/mlp_bot_hiddenlayer_2/ReluRelu+mlp_bot_layer/mlp_bot_hiddenlayer_2/BiasAdd*
T0*'
_output_shapes
:€€€€€€€€€@
ы
Umlp_bot_layer/mlp_bot_hiddenlayer_2/batch_normalization/gamma/part_0/Initializer/onesConst*
dtype0*
valueB@*  А?*
_output_shapes
:@*W
_classM
KIloc:@mlp_bot_layer/mlp_bot_hiddenlayer_2/batch_normalization/gamma/part_0
е
Dmlp_bot_layer/mlp_bot_hiddenlayer_2/batch_normalization/gamma/part_0
VariableV2*W
_classM
KIloc:@mlp_bot_layer/mlp_bot_hiddenlayer_2/batch_normalization/gamma/part_0*
_output_shapes
:@*
shape:@*
dtype0
р
Kmlp_bot_layer/mlp_bot_hiddenlayer_2/batch_normalization/gamma/part_0/AssignAssignDmlp_bot_layer/mlp_bot_hiddenlayer_2/batch_normalization/gamma/part_0Umlp_bot_layer/mlp_bot_hiddenlayer_2/batch_normalization/gamma/part_0/Initializer/ones*W
_classM
KIloc:@mlp_bot_layer/mlp_bot_hiddenlayer_2/batch_normalization/gamma/part_0*
T0*
_output_shapes
:@
Щ
Imlp_bot_layer/mlp_bot_hiddenlayer_2/batch_normalization/gamma/part_0/readIdentityDmlp_bot_layer/mlp_bot_hiddenlayer_2/batch_normalization/gamma/part_0*W
_classM
KIloc:@mlp_bot_layer/mlp_bot_hiddenlayer_2/batch_normalization/gamma/part_0*
T0*
_output_shapes
:@
ъ
Umlp_bot_layer/mlp_bot_hiddenlayer_2/batch_normalization/beta/part_0/Initializer/zerosConst*
valueB@*    *
_output_shapes
:@*V
_classL
JHloc:@mlp_bot_layer/mlp_bot_hiddenlayer_2/batch_normalization/beta/part_0*
dtype0
г
Cmlp_bot_layer/mlp_bot_hiddenlayer_2/batch_normalization/beta/part_0
VariableV2*V
_classL
JHloc:@mlp_bot_layer/mlp_bot_hiddenlayer_2/batch_normalization/beta/part_0*
shape:@*
_output_shapes
:@*
dtype0
н
Jmlp_bot_layer/mlp_bot_hiddenlayer_2/batch_normalization/beta/part_0/AssignAssignCmlp_bot_layer/mlp_bot_hiddenlayer_2/batch_normalization/beta/part_0Umlp_bot_layer/mlp_bot_hiddenlayer_2/batch_normalization/beta/part_0/Initializer/zeros*
T0*
_output_shapes
:@*V
_classL
JHloc:@mlp_bot_layer/mlp_bot_hiddenlayer_2/batch_normalization/beta/part_0
Ц
Hmlp_bot_layer/mlp_bot_hiddenlayer_2/batch_normalization/beta/part_0/readIdentityCmlp_bot_layer/mlp_bot_hiddenlayer_2/batch_normalization/beta/part_0*
T0*
_output_shapes
:@*V
_classL
JHloc:@mlp_bot_layer/mlp_bot_hiddenlayer_2/batch_normalization/beta/part_0
ъ
Umlp_bot_layer/mlp_bot_hiddenlayer_2/batch_normalization/moving_mean/Initializer/zerosConst*
dtype0*V
_classL
JHloc:@mlp_bot_layer/mlp_bot_hiddenlayer_2/batch_normalization/moving_mean*
valueB@*    *
_output_shapes
:@
г
Cmlp_bot_layer/mlp_bot_hiddenlayer_2/batch_normalization/moving_mean
VariableV2*V
_classL
JHloc:@mlp_bot_layer/mlp_bot_hiddenlayer_2/batch_normalization/moving_mean*
shape:@*
_output_shapes
:@*
dtype0
н
Jmlp_bot_layer/mlp_bot_hiddenlayer_2/batch_normalization/moving_mean/AssignAssignCmlp_bot_layer/mlp_bot_hiddenlayer_2/batch_normalization/moving_meanUmlp_bot_layer/mlp_bot_hiddenlayer_2/batch_normalization/moving_mean/Initializer/zeros*
_output_shapes
:@*
T0*V
_classL
JHloc:@mlp_bot_layer/mlp_bot_hiddenlayer_2/batch_normalization/moving_mean
Ц
Hmlp_bot_layer/mlp_bot_hiddenlayer_2/batch_normalization/moving_mean/readIdentityCmlp_bot_layer/mlp_bot_hiddenlayer_2/batch_normalization/moving_mean*V
_classL
JHloc:@mlp_bot_layer/mlp_bot_hiddenlayer_2/batch_normalization/moving_mean*
T0*
_output_shapes
:@
Б
Xmlp_bot_layer/mlp_bot_hiddenlayer_2/batch_normalization/moving_variance/Initializer/onesConst*
dtype0*
valueB@*  А?*
_output_shapes
:@*Z
_classP
NLloc:@mlp_bot_layer/mlp_bot_hiddenlayer_2/batch_normalization/moving_variance
л
Gmlp_bot_layer/mlp_bot_hiddenlayer_2/batch_normalization/moving_variance
VariableV2*
shape:@*
_output_shapes
:@*
dtype0*Z
_classP
NLloc:@mlp_bot_layer/mlp_bot_hiddenlayer_2/batch_normalization/moving_variance
ь
Nmlp_bot_layer/mlp_bot_hiddenlayer_2/batch_normalization/moving_variance/AssignAssignGmlp_bot_layer/mlp_bot_hiddenlayer_2/batch_normalization/moving_varianceXmlp_bot_layer/mlp_bot_hiddenlayer_2/batch_normalization/moving_variance/Initializer/ones*
_output_shapes
:@*
T0*Z
_classP
NLloc:@mlp_bot_layer/mlp_bot_hiddenlayer_2/batch_normalization/moving_variance
Ґ
Lmlp_bot_layer/mlp_bot_hiddenlayer_2/batch_normalization/moving_variance/readIdentityGmlp_bot_layer/mlp_bot_hiddenlayer_2/batch_normalization/moving_variance*Z
_classP
NLloc:@mlp_bot_layer/mlp_bot_hiddenlayer_2/batch_normalization/moving_variance*
T0*
_output_shapes
:@
†
Vmlp_bot_layer/mlp_bot_hiddenlayer_2/batch_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
valueB: *
dtype0
И
Dmlp_bot_layer/mlp_bot_hiddenlayer_2/batch_normalization/moments/meanMean(mlp_bot_layer/mlp_bot_hiddenlayer_2/ReluVmlp_bot_layer/mlp_bot_hiddenlayer_2/batch_normalization/moments/mean/reduction_indices*
_output_shapes

:@*
	keep_dims(*
T0
Ћ
Lmlp_bot_layer/mlp_bot_hiddenlayer_2/batch_normalization/moments/StopGradientStopGradientDmlp_bot_layer/mlp_bot_hiddenlayer_2/batch_normalization/moments/mean*
T0*
_output_shapes

:@
Р
Qmlp_bot_layer/mlp_bot_hiddenlayer_2/batch_normalization/moments/SquaredDifferenceSquaredDifference(mlp_bot_layer/mlp_bot_hiddenlayer_2/ReluLmlp_bot_layer/mlp_bot_hiddenlayer_2/batch_normalization/moments/StopGradient*
T0*'
_output_shapes
:€€€€€€€€€@
§
Zmlp_bot_layer/mlp_bot_hiddenlayer_2/batch_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
є
Hmlp_bot_layer/mlp_bot_hiddenlayer_2/batch_normalization/moments/varianceMeanQmlp_bot_layer/mlp_bot_hiddenlayer_2/batch_normalization/moments/SquaredDifferenceZmlp_bot_layer/mlp_bot_hiddenlayer_2/batch_normalization/moments/variance/reduction_indices*
T0*
_output_shapes

:@*
	keep_dims(
‘
Gmlp_bot_layer/mlp_bot_hiddenlayer_2/batch_normalization/moments/SqueezeSqueezeDmlp_bot_layer/mlp_bot_hiddenlayer_2/batch_normalization/moments/mean*
T0*
squeeze_dims
 *
_output_shapes
:@
Џ
Imlp_bot_layer/mlp_bot_hiddenlayer_2/batch_normalization/moments/Squeeze_1SqueezeHmlp_bot_layer/mlp_bot_hiddenlayer_2/batch_normalization/moments/variance*
squeeze_dims
 *
_output_shapes
:@*
T0
к
Mmlp_bot_layer/mlp_bot_hiddenlayer_2/batch_normalization/AssignMovingAvg/decayConst*
_output_shapes
: *
valueB
 *
„#<*
dtype0*V
_classL
JHloc:@mlp_bot_layer/mlp_bot_hiddenlayer_2/batch_normalization/moving_mean
в
Kmlp_bot_layer/mlp_bot_hiddenlayer_2/batch_normalization/AssignMovingAvg/subSubHmlp_bot_layer/mlp_bot_hiddenlayer_2/batch_normalization/moving_mean/readGmlp_bot_layer/mlp_bot_hiddenlayer_2/batch_normalization/moments/Squeeze*
_output_shapes
:@*V
_classL
JHloc:@mlp_bot_layer/mlp_bot_hiddenlayer_2/batch_normalization/moving_mean*
T0
л
Kmlp_bot_layer/mlp_bot_hiddenlayer_2/batch_normalization/AssignMovingAvg/mulMulKmlp_bot_layer/mlp_bot_hiddenlayer_2/batch_normalization/AssignMovingAvg/subMmlp_bot_layer/mlp_bot_hiddenlayer_2/batch_normalization/AssignMovingAvg/decay*V
_classL
JHloc:@mlp_bot_layer/mlp_bot_hiddenlayer_2/batch_normalization/moving_mean*
T0*
_output_shapes
:@
г
Gmlp_bot_layer/mlp_bot_hiddenlayer_2/batch_normalization/AssignMovingAvg	AssignSubCmlp_bot_layer/mlp_bot_hiddenlayer_2/batch_normalization/moving_meanKmlp_bot_layer/mlp_bot_hiddenlayer_2/batch_normalization/AssignMovingAvg/mul*
_output_shapes
:@*
T0*V
_classL
JHloc:@mlp_bot_layer/mlp_bot_hiddenlayer_2/batch_normalization/moving_mean
р
Omlp_bot_layer/mlp_bot_hiddenlayer_2/batch_normalization/AssignMovingAvg_1/decayConst*Z
_classP
NLloc:@mlp_bot_layer/mlp_bot_hiddenlayer_2/batch_normalization/moving_variance*
dtype0*
_output_shapes
: *
valueB
 *
„#<
о
Mmlp_bot_layer/mlp_bot_hiddenlayer_2/batch_normalization/AssignMovingAvg_1/subSubLmlp_bot_layer/mlp_bot_hiddenlayer_2/batch_normalization/moving_variance/readImlp_bot_layer/mlp_bot_hiddenlayer_2/batch_normalization/moments/Squeeze_1*
_output_shapes
:@*Z
_classP
NLloc:@mlp_bot_layer/mlp_bot_hiddenlayer_2/batch_normalization/moving_variance*
T0
х
Mmlp_bot_layer/mlp_bot_hiddenlayer_2/batch_normalization/AssignMovingAvg_1/mulMulMmlp_bot_layer/mlp_bot_hiddenlayer_2/batch_normalization/AssignMovingAvg_1/subOmlp_bot_layer/mlp_bot_hiddenlayer_2/batch_normalization/AssignMovingAvg_1/decay*
T0*Z
_classP
NLloc:@mlp_bot_layer/mlp_bot_hiddenlayer_2/batch_normalization/moving_variance*
_output_shapes
:@
п
Imlp_bot_layer/mlp_bot_hiddenlayer_2/batch_normalization/AssignMovingAvg_1	AssignSubGmlp_bot_layer/mlp_bot_hiddenlayer_2/batch_normalization/moving_varianceMmlp_bot_layer/mlp_bot_hiddenlayer_2/batch_normalization/AssignMovingAvg_1/mul*
_output_shapes
:@*Z
_classP
NLloc:@mlp_bot_layer/mlp_bot_hiddenlayer_2/batch_normalization/moving_variance*
T0
Ј
<mlp_bot_layer/mlp_bot_hiddenlayer_2/batch_normalization/betaIdentityHmlp_bot_layer/mlp_bot_hiddenlayer_2/batch_normalization/beta/part_0/read*
T0*
_output_shapes
:@
є
=mlp_bot_layer/mlp_bot_hiddenlayer_2/batch_normalization/gammaIdentityImlp_bot_layer/mlp_bot_hiddenlayer_2/batch_normalization/gamma/part_0/read*
T0*
_output_shapes
:@
М
Gmlp_bot_layer/mlp_bot_hiddenlayer_2/batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
valueB
 *oГ:*
dtype0
З
Emlp_bot_layer/mlp_bot_hiddenlayer_2/batch_normalization/batchnorm/addAddV2Imlp_bot_layer/mlp_bot_hiddenlayer_2/batch_normalization/moments/Squeeze_1Gmlp_bot_layer/mlp_bot_hiddenlayer_2/batch_normalization/batchnorm/add/y*
_output_shapes
:@*
T0
Љ
Gmlp_bot_layer/mlp_bot_hiddenlayer_2/batch_normalization/batchnorm/RsqrtRsqrtEmlp_bot_layer/mlp_bot_hiddenlayer_2/batch_normalization/batchnorm/add*
T0*
_output_shapes
:@
щ
Emlp_bot_layer/mlp_bot_hiddenlayer_2/batch_normalization/batchnorm/mulMulGmlp_bot_layer/mlp_bot_hiddenlayer_2/batch_normalization/batchnorm/Rsqrt=mlp_bot_layer/mlp_bot_hiddenlayer_2/batch_normalization/gamma*
_output_shapes
:@*
T0
с
Gmlp_bot_layer/mlp_bot_hiddenlayer_2/batch_normalization/batchnorm/mul_1Mul(mlp_bot_layer/mlp_bot_hiddenlayer_2/ReluEmlp_bot_layer/mlp_bot_hiddenlayer_2/batch_normalization/batchnorm/mul*
T0*'
_output_shapes
:€€€€€€€€€@
Г
Gmlp_bot_layer/mlp_bot_hiddenlayer_2/batch_normalization/batchnorm/mul_2MulGmlp_bot_layer/mlp_bot_hiddenlayer_2/batch_normalization/moments/SqueezeEmlp_bot_layer/mlp_bot_hiddenlayer_2/batch_normalization/batchnorm/mul*
T0*
_output_shapes
:@
ш
Emlp_bot_layer/mlp_bot_hiddenlayer_2/batch_normalization/batchnorm/subSub<mlp_bot_layer/mlp_bot_hiddenlayer_2/batch_normalization/betaGmlp_bot_layer/mlp_bot_hiddenlayer_2/batch_normalization/batchnorm/mul_2*
_output_shapes
:@*
T0
Т
Gmlp_bot_layer/mlp_bot_hiddenlayer_2/batch_normalization/batchnorm/add_1AddV2Gmlp_bot_layer/mlp_bot_hiddenlayer_2/batch_normalization/batchnorm/mul_1Emlp_bot_layer/mlp_bot_hiddenlayer_2/batch_normalization/batchnorm/sub*'
_output_shapes
:€€€€€€€€€@*
T0
Є
6mlp_bot_layer/mlp_bot_hiddenlayer_2/zero_fraction/SizeSizeGmlp_bot_layer/mlp_bot_hiddenlayer_2/batch_normalization/batchnorm/add_1*
_output_shapes
: *
T0*
out_type0	
Г
=mlp_bot_layer/mlp_bot_hiddenlayer_2/zero_fraction/LessEqual/yConst*
dtype0	*
_output_shapes
: *
valueB	 R€€€€
а
;mlp_bot_layer/mlp_bot_hiddenlayer_2/zero_fraction/LessEqual	LessEqual6mlp_bot_layer/mlp_bot_hiddenlayer_2/zero_fraction/Size=mlp_bot_layer/mlp_bot_hiddenlayer_2/zero_fraction/LessEqual/y*
_output_shapes
: *
T0	
д
=mlp_bot_layer/mlp_bot_hiddenlayer_2/zero_fraction/cond/SwitchSwitch;mlp_bot_layer/mlp_bot_hiddenlayer_2/zero_fraction/LessEqual;mlp_bot_layer/mlp_bot_hiddenlayer_2/zero_fraction/LessEqual*
T0
*
_output_shapes
: : 
≠
?mlp_bot_layer/mlp_bot_hiddenlayer_2/zero_fraction/cond/switch_tIdentity?mlp_bot_layer/mlp_bot_hiddenlayer_2/zero_fraction/cond/Switch:1*
T0
*
_output_shapes
: 
Ђ
?mlp_bot_layer/mlp_bot_hiddenlayer_2/zero_fraction/cond/switch_fIdentity=mlp_bot_layer/mlp_bot_hiddenlayer_2/zero_fraction/cond/Switch*
T0
*
_output_shapes
: 
®
>mlp_bot_layer/mlp_bot_hiddenlayer_2/zero_fraction/cond/pred_idIdentity;mlp_bot_layer/mlp_bot_hiddenlayer_2/zero_fraction/LessEqual*
_output_shapes
: *
T0

—
Jmlp_bot_layer/mlp_bot_hiddenlayer_2/zero_fraction/cond/count_nonzero/zerosConst@^mlp_bot_layer/mlp_bot_hiddenlayer_2/zero_fraction/cond/switch_t*
dtype0*
_output_shapes
: *
valueB
 *    
ѓ
Mmlp_bot_layer/mlp_bot_hiddenlayer_2/zero_fraction/cond/count_nonzero/NotEqualNotEqualVmlp_bot_layer/mlp_bot_hiddenlayer_2/zero_fraction/cond/count_nonzero/NotEqual/Switch:1Jmlp_bot_layer/mlp_bot_hiddenlayer_2/zero_fraction/cond/count_nonzero/zeros*'
_output_shapes
:€€€€€€€€€@*
T0
И
Tmlp_bot_layer/mlp_bot_hiddenlayer_2/zero_fraction/cond/count_nonzero/NotEqual/SwitchSwitchGmlp_bot_layer/mlp_bot_hiddenlayer_2/batch_normalization/batchnorm/add_1>mlp_bot_layer/mlp_bot_hiddenlayer_2/zero_fraction/cond/pred_id*Z
_classP
NLloc:@mlp_bot_layer/mlp_bot_hiddenlayer_2/batch_normalization/batchnorm/add_1*
T0*:
_output_shapes(
&:€€€€€€€€€@:€€€€€€€€€@
б
Imlp_bot_layer/mlp_bot_hiddenlayer_2/zero_fraction/cond/count_nonzero/CastCastMmlp_bot_layer/mlp_bot_hiddenlayer_2/zero_fraction/cond/count_nonzero/NotEqual*

SrcT0
*

DstT0*'
_output_shapes
:€€€€€€€€€@
Ё
Jmlp_bot_layer/mlp_bot_hiddenlayer_2/zero_fraction/cond/count_nonzero/ConstConst@^mlp_bot_layer/mlp_bot_hiddenlayer_2/zero_fraction/cond/switch_t*
valueB"       *
_output_shapes
:*
dtype0
С
Rmlp_bot_layer/mlp_bot_hiddenlayer_2/zero_fraction/cond/count_nonzero/nonzero_countSumImlp_bot_layer/mlp_bot_hiddenlayer_2/zero_fraction/cond/count_nonzero/CastJmlp_bot_layer/mlp_bot_hiddenlayer_2/zero_fraction/cond/count_nonzero/Const*
T0*
_output_shapes
: 
«
;mlp_bot_layer/mlp_bot_hiddenlayer_2/zero_fraction/cond/CastCastRmlp_bot_layer/mlp_bot_hiddenlayer_2/zero_fraction/cond/count_nonzero/nonzero_count*

SrcT0*
_output_shapes
: *

DstT0	
”
Lmlp_bot_layer/mlp_bot_hiddenlayer_2/zero_fraction/cond/count_nonzero_1/zerosConst@^mlp_bot_layer/mlp_bot_hiddenlayer_2/zero_fraction/cond/switch_f*
_output_shapes
: *
valueB
 *    *
dtype0
≥
Omlp_bot_layer/mlp_bot_hiddenlayer_2/zero_fraction/cond/count_nonzero_1/NotEqualNotEqualVmlp_bot_layer/mlp_bot_hiddenlayer_2/zero_fraction/cond/count_nonzero_1/NotEqual/SwitchLmlp_bot_layer/mlp_bot_hiddenlayer_2/zero_fraction/cond/count_nonzero_1/zeros*'
_output_shapes
:€€€€€€€€€@*
T0
К
Vmlp_bot_layer/mlp_bot_hiddenlayer_2/zero_fraction/cond/count_nonzero_1/NotEqual/SwitchSwitchGmlp_bot_layer/mlp_bot_hiddenlayer_2/batch_normalization/batchnorm/add_1>mlp_bot_layer/mlp_bot_hiddenlayer_2/zero_fraction/cond/pred_id*
T0*:
_output_shapes(
&:€€€€€€€€€@:€€€€€€€€€@*Z
_classP
NLloc:@mlp_bot_layer/mlp_bot_hiddenlayer_2/batch_normalization/batchnorm/add_1
е
Kmlp_bot_layer/mlp_bot_hiddenlayer_2/zero_fraction/cond/count_nonzero_1/CastCastOmlp_bot_layer/mlp_bot_hiddenlayer_2/zero_fraction/cond/count_nonzero_1/NotEqual*

SrcT0
*

DstT0	*'
_output_shapes
:€€€€€€€€€@
я
Lmlp_bot_layer/mlp_bot_hiddenlayer_2/zero_fraction/cond/count_nonzero_1/ConstConst@^mlp_bot_layer/mlp_bot_hiddenlayer_2/zero_fraction/cond/switch_f*
valueB"       *
_output_shapes
:*
dtype0
Ч
Tmlp_bot_layer/mlp_bot_hiddenlayer_2/zero_fraction/cond/count_nonzero_1/nonzero_countSumKmlp_bot_layer/mlp_bot_hiddenlayer_2/zero_fraction/cond/count_nonzero_1/CastLmlp_bot_layer/mlp_bot_hiddenlayer_2/zero_fraction/cond/count_nonzero_1/Const*
_output_shapes
: *
T0	
Д
<mlp_bot_layer/mlp_bot_hiddenlayer_2/zero_fraction/cond/MergeMergeTmlp_bot_layer/mlp_bot_hiddenlayer_2/zero_fraction/cond/count_nonzero_1/nonzero_count;mlp_bot_layer/mlp_bot_hiddenlayer_2/zero_fraction/cond/Cast*
T0	*
N*
_output_shapes
: : 
ж
Hmlp_bot_layer/mlp_bot_hiddenlayer_2/zero_fraction/counts_to_fraction/subSub6mlp_bot_layer/mlp_bot_hiddenlayer_2/zero_fraction/Size<mlp_bot_layer/mlp_bot_hiddenlayer_2/zero_fraction/cond/Merge*
_output_shapes
: *
T0	
Ћ
Imlp_bot_layer/mlp_bot_hiddenlayer_2/zero_fraction/counts_to_fraction/CastCastHmlp_bot_layer/mlp_bot_hiddenlayer_2/zero_fraction/counts_to_fraction/sub*

SrcT0	*

DstT0*
_output_shapes
: 
ї
Kmlp_bot_layer/mlp_bot_hiddenlayer_2/zero_fraction/counts_to_fraction/Cast_1Cast6mlp_bot_layer/mlp_bot_hiddenlayer_2/zero_fraction/Size*

SrcT0	*
_output_shapes
: *

DstT0
Р
Lmlp_bot_layer/mlp_bot_hiddenlayer_2/zero_fraction/counts_to_fraction/truedivRealDivImlp_bot_layer/mlp_bot_hiddenlayer_2/zero_fraction/counts_to_fraction/CastKmlp_bot_layer/mlp_bot_hiddenlayer_2/zero_fraction/counts_to_fraction/Cast_1*
_output_shapes
: *
T0
µ
:mlp_bot_layer/mlp_bot_hiddenlayer_2/zero_fraction/fractionIdentityLmlp_bot_layer/mlp_bot_hiddenlayer_2/zero_fraction/counts_to_fraction/truediv*
T0*
_output_shapes
: 
Д
dmlp_bot_layer/mlp_bot_hiddenlayer_2/mlp_bot_layer/mlp_bot_hiddenlayer_2/fraction_of_zero_values/tagsConst*
dtype0*p
valuegBe B_mlp_bot_layer/mlp_bot_hiddenlayer_2/mlp_bot_layer/mlp_bot_hiddenlayer_2/fraction_of_zero_values*
_output_shapes
: 
≥
_mlp_bot_layer/mlp_bot_hiddenlayer_2/mlp_bot_layer/mlp_bot_hiddenlayer_2/fraction_of_zero_valuesScalarSummarydmlp_bot_layer/mlp_bot_hiddenlayer_2/mlp_bot_layer/mlp_bot_hiddenlayer_2/fraction_of_zero_values/tags:mlp_bot_layer/mlp_bot_hiddenlayer_2/zero_fraction/fraction*
_output_shapes
: *
T0
й
Vmlp_bot_layer/mlp_bot_hiddenlayer_2/mlp_bot_layer/mlp_bot_hiddenlayer_2/activation/tagConst*
dtype0*
_output_shapes
: *c
valueZBX BRmlp_bot_layer/mlp_bot_hiddenlayer_2/mlp_bot_layer/mlp_bot_hiddenlayer_2/activation
Я
Rmlp_bot_layer/mlp_bot_hiddenlayer_2/mlp_bot_layer/mlp_bot_hiddenlayer_2/activationHistogramSummaryVmlp_bot_layer/mlp_bot_hiddenlayer_2/mlp_bot_layer/mlp_bot_hiddenlayer_2/activation/tagGmlp_bot_layer/mlp_bot_hiddenlayer_2/batch_normalization/batchnorm/add_1*
_output_shapes
: 
й
Rmlp_bot_layer/mlp_bot_hiddenlayer_3/kernel/part_0/Initializer/random_uniform/shapeConst*D
_class:
86loc:@mlp_bot_layer/mlp_bot_hiddenlayer_3/kernel/part_0*
_output_shapes
:*
dtype0*
valueB"@      
џ
Pmlp_bot_layer/mlp_bot_hiddenlayer_3/kernel/part_0/Initializer/random_uniform/minConst*D
_class:
86loc:@mlp_bot_layer/mlp_bot_hiddenlayer_3/kernel/part_0*
valueB
 *М7МЊ*
_output_shapes
: *
dtype0
џ
Pmlp_bot_layer/mlp_bot_hiddenlayer_3/kernel/part_0/Initializer/random_uniform/maxConst*D
_class:
86loc:@mlp_bot_layer/mlp_bot_hiddenlayer_3/kernel/part_0*
_output_shapes
: *
valueB
 *М7М>*
dtype0
ї
Zmlp_bot_layer/mlp_bot_hiddenlayer_3/kernel/part_0/Initializer/random_uniform/RandomUniformRandomUniformRmlp_bot_layer/mlp_bot_hiddenlayer_3/kernel/part_0/Initializer/random_uniform/shape*
dtype0*
_output_shapes

:@*
T0*D
_class:
86loc:@mlp_bot_layer/mlp_bot_hiddenlayer_3/kernel/part_0
в
Pmlp_bot_layer/mlp_bot_hiddenlayer_3/kernel/part_0/Initializer/random_uniform/subSubPmlp_bot_layer/mlp_bot_hiddenlayer_3/kernel/part_0/Initializer/random_uniform/maxPmlp_bot_layer/mlp_bot_hiddenlayer_3/kernel/part_0/Initializer/random_uniform/min*D
_class:
86loc:@mlp_bot_layer/mlp_bot_hiddenlayer_3/kernel/part_0*
T0*
_output_shapes
: 
ф
Pmlp_bot_layer/mlp_bot_hiddenlayer_3/kernel/part_0/Initializer/random_uniform/mulMulZmlp_bot_layer/mlp_bot_hiddenlayer_3/kernel/part_0/Initializer/random_uniform/RandomUniformPmlp_bot_layer/mlp_bot_hiddenlayer_3/kernel/part_0/Initializer/random_uniform/sub*
T0*
_output_shapes

:@*D
_class:
86loc:@mlp_bot_layer/mlp_bot_hiddenlayer_3/kernel/part_0
ж
Lmlp_bot_layer/mlp_bot_hiddenlayer_3/kernel/part_0/Initializer/random_uniformAddPmlp_bot_layer/mlp_bot_hiddenlayer_3/kernel/part_0/Initializer/random_uniform/mulPmlp_bot_layer/mlp_bot_hiddenlayer_3/kernel/part_0/Initializer/random_uniform/min*
T0*D
_class:
86loc:@mlp_bot_layer/mlp_bot_hiddenlayer_3/kernel/part_0*
_output_shapes

:@
«
1mlp_bot_layer/mlp_bot_hiddenlayer_3/kernel/part_0
VariableV2*
shape
:@*D
_class:
86loc:@mlp_bot_layer/mlp_bot_hiddenlayer_3/kernel/part_0*
dtype0*
_output_shapes

:@
≤
8mlp_bot_layer/mlp_bot_hiddenlayer_3/kernel/part_0/AssignAssign1mlp_bot_layer/mlp_bot_hiddenlayer_3/kernel/part_0Lmlp_bot_layer/mlp_bot_hiddenlayer_3/kernel/part_0/Initializer/random_uniform*
T0*
_output_shapes

:@*D
_class:
86loc:@mlp_bot_layer/mlp_bot_hiddenlayer_3/kernel/part_0
д
6mlp_bot_layer/mlp_bot_hiddenlayer_3/kernel/part_0/readIdentity1mlp_bot_layer/mlp_bot_hiddenlayer_3/kernel/part_0*
_output_shapes

:@*D
_class:
86loc:@mlp_bot_layer/mlp_bot_hiddenlayer_3/kernel/part_0*
T0
“
Amlp_bot_layer/mlp_bot_hiddenlayer_3/bias/part_0/Initializer/zerosConst*
valueB*    *
_output_shapes
:*B
_class8
64loc:@mlp_bot_layer/mlp_bot_hiddenlayer_3/bias/part_0*
dtype0
ї
/mlp_bot_layer/mlp_bot_hiddenlayer_3/bias/part_0
VariableV2*
_output_shapes
:*
shape:*B
_class8
64loc:@mlp_bot_layer/mlp_bot_hiddenlayer_3/bias/part_0*
dtype0
Э
6mlp_bot_layer/mlp_bot_hiddenlayer_3/bias/part_0/AssignAssign/mlp_bot_layer/mlp_bot_hiddenlayer_3/bias/part_0Amlp_bot_layer/mlp_bot_hiddenlayer_3/bias/part_0/Initializer/zeros*B
_class8
64loc:@mlp_bot_layer/mlp_bot_hiddenlayer_3/bias/part_0*
_output_shapes
:*
T0
Џ
4mlp_bot_layer/mlp_bot_hiddenlayer_3/bias/part_0/readIdentity/mlp_bot_layer/mlp_bot_hiddenlayer_3/bias/part_0*
_output_shapes
:*B
_class8
64loc:@mlp_bot_layer/mlp_bot_hiddenlayer_3/bias/part_0*
T0
Ч
*mlp_bot_layer/mlp_bot_hiddenlayer_3/kernelIdentity6mlp_bot_layer/mlp_bot_hiddenlayer_3/kernel/part_0/read*
_output_shapes

:@*
T0
џ
*mlp_bot_layer/mlp_bot_hiddenlayer_3/MatMulMatMulGmlp_bot_layer/mlp_bot_hiddenlayer_2/batch_normalization/batchnorm/add_1*mlp_bot_layer/mlp_bot_hiddenlayer_3/kernel*'
_output_shapes
:€€€€€€€€€*
T0
П
(mlp_bot_layer/mlp_bot_hiddenlayer_3/biasIdentity4mlp_bot_layer/mlp_bot_hiddenlayer_3/bias/part_0/read*
T0*
_output_shapes
:
Њ
+mlp_bot_layer/mlp_bot_hiddenlayer_3/BiasAddBiasAdd*mlp_bot_layer/mlp_bot_hiddenlayer_3/MatMul(mlp_bot_layer/mlp_bot_hiddenlayer_3/bias*'
_output_shapes
:€€€€€€€€€*
T0
П
(mlp_bot_layer/mlp_bot_hiddenlayer_3/ReluRelu+mlp_bot_layer/mlp_bot_hiddenlayer_3/BiasAdd*'
_output_shapes
:€€€€€€€€€*
T0
ы
Umlp_bot_layer/mlp_bot_hiddenlayer_3/batch_normalization/gamma/part_0/Initializer/onesConst*W
_classM
KIloc:@mlp_bot_layer/mlp_bot_hiddenlayer_3/batch_normalization/gamma/part_0*
valueB*  А?*
dtype0*
_output_shapes
:
е
Dmlp_bot_layer/mlp_bot_hiddenlayer_3/batch_normalization/gamma/part_0
VariableV2*W
_classM
KIloc:@mlp_bot_layer/mlp_bot_hiddenlayer_3/batch_normalization/gamma/part_0*
shape:*
_output_shapes
:*
dtype0
р
Kmlp_bot_layer/mlp_bot_hiddenlayer_3/batch_normalization/gamma/part_0/AssignAssignDmlp_bot_layer/mlp_bot_hiddenlayer_3/batch_normalization/gamma/part_0Umlp_bot_layer/mlp_bot_hiddenlayer_3/batch_normalization/gamma/part_0/Initializer/ones*W
_classM
KIloc:@mlp_bot_layer/mlp_bot_hiddenlayer_3/batch_normalization/gamma/part_0*
T0*
_output_shapes
:
Щ
Imlp_bot_layer/mlp_bot_hiddenlayer_3/batch_normalization/gamma/part_0/readIdentityDmlp_bot_layer/mlp_bot_hiddenlayer_3/batch_normalization/gamma/part_0*W
_classM
KIloc:@mlp_bot_layer/mlp_bot_hiddenlayer_3/batch_normalization/gamma/part_0*
T0*
_output_shapes
:
ъ
Umlp_bot_layer/mlp_bot_hiddenlayer_3/batch_normalization/beta/part_0/Initializer/zerosConst*
dtype0*V
_classL
JHloc:@mlp_bot_layer/mlp_bot_hiddenlayer_3/batch_normalization/beta/part_0*
valueB*    *
_output_shapes
:
г
Cmlp_bot_layer/mlp_bot_hiddenlayer_3/batch_normalization/beta/part_0
VariableV2*
_output_shapes
:*V
_classL
JHloc:@mlp_bot_layer/mlp_bot_hiddenlayer_3/batch_normalization/beta/part_0*
shape:*
dtype0
н
Jmlp_bot_layer/mlp_bot_hiddenlayer_3/batch_normalization/beta/part_0/AssignAssignCmlp_bot_layer/mlp_bot_hiddenlayer_3/batch_normalization/beta/part_0Umlp_bot_layer/mlp_bot_hiddenlayer_3/batch_normalization/beta/part_0/Initializer/zeros*V
_classL
JHloc:@mlp_bot_layer/mlp_bot_hiddenlayer_3/batch_normalization/beta/part_0*
_output_shapes
:*
T0
Ц
Hmlp_bot_layer/mlp_bot_hiddenlayer_3/batch_normalization/beta/part_0/readIdentityCmlp_bot_layer/mlp_bot_hiddenlayer_3/batch_normalization/beta/part_0*
_output_shapes
:*V
_classL
JHloc:@mlp_bot_layer/mlp_bot_hiddenlayer_3/batch_normalization/beta/part_0*
T0
ъ
Umlp_bot_layer/mlp_bot_hiddenlayer_3/batch_normalization/moving_mean/Initializer/zerosConst*
_output_shapes
:*
valueB*    *V
_classL
JHloc:@mlp_bot_layer/mlp_bot_hiddenlayer_3/batch_normalization/moving_mean*
dtype0
г
Cmlp_bot_layer/mlp_bot_hiddenlayer_3/batch_normalization/moving_mean
VariableV2*
shape:*
_output_shapes
:*V
_classL
JHloc:@mlp_bot_layer/mlp_bot_hiddenlayer_3/batch_normalization/moving_mean*
dtype0
н
Jmlp_bot_layer/mlp_bot_hiddenlayer_3/batch_normalization/moving_mean/AssignAssignCmlp_bot_layer/mlp_bot_hiddenlayer_3/batch_normalization/moving_meanUmlp_bot_layer/mlp_bot_hiddenlayer_3/batch_normalization/moving_mean/Initializer/zeros*V
_classL
JHloc:@mlp_bot_layer/mlp_bot_hiddenlayer_3/batch_normalization/moving_mean*
T0*
_output_shapes
:
Ц
Hmlp_bot_layer/mlp_bot_hiddenlayer_3/batch_normalization/moving_mean/readIdentityCmlp_bot_layer/mlp_bot_hiddenlayer_3/batch_normalization/moving_mean*
T0*V
_classL
JHloc:@mlp_bot_layer/mlp_bot_hiddenlayer_3/batch_normalization/moving_mean*
_output_shapes
:
Б
Xmlp_bot_layer/mlp_bot_hiddenlayer_3/batch_normalization/moving_variance/Initializer/onesConst*
dtype0*
valueB*  А?*Z
_classP
NLloc:@mlp_bot_layer/mlp_bot_hiddenlayer_3/batch_normalization/moving_variance*
_output_shapes
:
л
Gmlp_bot_layer/mlp_bot_hiddenlayer_3/batch_normalization/moving_variance
VariableV2*
_output_shapes
:*
shape:*Z
_classP
NLloc:@mlp_bot_layer/mlp_bot_hiddenlayer_3/batch_normalization/moving_variance*
dtype0
ь
Nmlp_bot_layer/mlp_bot_hiddenlayer_3/batch_normalization/moving_variance/AssignAssignGmlp_bot_layer/mlp_bot_hiddenlayer_3/batch_normalization/moving_varianceXmlp_bot_layer/mlp_bot_hiddenlayer_3/batch_normalization/moving_variance/Initializer/ones*
_output_shapes
:*Z
_classP
NLloc:@mlp_bot_layer/mlp_bot_hiddenlayer_3/batch_normalization/moving_variance*
T0
Ґ
Lmlp_bot_layer/mlp_bot_hiddenlayer_3/batch_normalization/moving_variance/readIdentityGmlp_bot_layer/mlp_bot_hiddenlayer_3/batch_normalization/moving_variance*
_output_shapes
:*
T0*Z
_classP
NLloc:@mlp_bot_layer/mlp_bot_hiddenlayer_3/batch_normalization/moving_variance
†
Vmlp_bot_layer/mlp_bot_hiddenlayer_3/batch_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
valueB: *
dtype0
И
Dmlp_bot_layer/mlp_bot_hiddenlayer_3/batch_normalization/moments/meanMean(mlp_bot_layer/mlp_bot_hiddenlayer_3/ReluVmlp_bot_layer/mlp_bot_hiddenlayer_3/batch_normalization/moments/mean/reduction_indices*
_output_shapes

:*
	keep_dims(*
T0
Ћ
Lmlp_bot_layer/mlp_bot_hiddenlayer_3/batch_normalization/moments/StopGradientStopGradientDmlp_bot_layer/mlp_bot_hiddenlayer_3/batch_normalization/moments/mean*
T0*
_output_shapes

:
Р
Qmlp_bot_layer/mlp_bot_hiddenlayer_3/batch_normalization/moments/SquaredDifferenceSquaredDifference(mlp_bot_layer/mlp_bot_hiddenlayer_3/ReluLmlp_bot_layer/mlp_bot_hiddenlayer_3/batch_normalization/moments/StopGradient*'
_output_shapes
:€€€€€€€€€*
T0
§
Zmlp_bot_layer/mlp_bot_hiddenlayer_3/batch_normalization/moments/variance/reduction_indicesConst*
dtype0*
valueB: *
_output_shapes
:
є
Hmlp_bot_layer/mlp_bot_hiddenlayer_3/batch_normalization/moments/varianceMeanQmlp_bot_layer/mlp_bot_hiddenlayer_3/batch_normalization/moments/SquaredDifferenceZmlp_bot_layer/mlp_bot_hiddenlayer_3/batch_normalization/moments/variance/reduction_indices*
	keep_dims(*
T0*
_output_shapes

:
‘
Gmlp_bot_layer/mlp_bot_hiddenlayer_3/batch_normalization/moments/SqueezeSqueezeDmlp_bot_layer/mlp_bot_hiddenlayer_3/batch_normalization/moments/mean*
squeeze_dims
 *
T0*
_output_shapes
:
Џ
Imlp_bot_layer/mlp_bot_hiddenlayer_3/batch_normalization/moments/Squeeze_1SqueezeHmlp_bot_layer/mlp_bot_hiddenlayer_3/batch_normalization/moments/variance*
squeeze_dims
 *
T0*
_output_shapes
:
к
Mmlp_bot_layer/mlp_bot_hiddenlayer_3/batch_normalization/AssignMovingAvg/decayConst*V
_classL
JHloc:@mlp_bot_layer/mlp_bot_hiddenlayer_3/batch_normalization/moving_mean*
valueB
 *
„#<*
dtype0*
_output_shapes
: 
в
Kmlp_bot_layer/mlp_bot_hiddenlayer_3/batch_normalization/AssignMovingAvg/subSubHmlp_bot_layer/mlp_bot_hiddenlayer_3/batch_normalization/moving_mean/readGmlp_bot_layer/mlp_bot_hiddenlayer_3/batch_normalization/moments/Squeeze*
T0*V
_classL
JHloc:@mlp_bot_layer/mlp_bot_hiddenlayer_3/batch_normalization/moving_mean*
_output_shapes
:
л
Kmlp_bot_layer/mlp_bot_hiddenlayer_3/batch_normalization/AssignMovingAvg/mulMulKmlp_bot_layer/mlp_bot_hiddenlayer_3/batch_normalization/AssignMovingAvg/subMmlp_bot_layer/mlp_bot_hiddenlayer_3/batch_normalization/AssignMovingAvg/decay*
_output_shapes
:*
T0*V
_classL
JHloc:@mlp_bot_layer/mlp_bot_hiddenlayer_3/batch_normalization/moving_mean
г
Gmlp_bot_layer/mlp_bot_hiddenlayer_3/batch_normalization/AssignMovingAvg	AssignSubCmlp_bot_layer/mlp_bot_hiddenlayer_3/batch_normalization/moving_meanKmlp_bot_layer/mlp_bot_hiddenlayer_3/batch_normalization/AssignMovingAvg/mul*
_output_shapes
:*V
_classL
JHloc:@mlp_bot_layer/mlp_bot_hiddenlayer_3/batch_normalization/moving_mean*
T0
р
Omlp_bot_layer/mlp_bot_hiddenlayer_3/batch_normalization/AssignMovingAvg_1/decayConst*
valueB
 *
„#<*
dtype0*Z
_classP
NLloc:@mlp_bot_layer/mlp_bot_hiddenlayer_3/batch_normalization/moving_variance*
_output_shapes
: 
о
Mmlp_bot_layer/mlp_bot_hiddenlayer_3/batch_normalization/AssignMovingAvg_1/subSubLmlp_bot_layer/mlp_bot_hiddenlayer_3/batch_normalization/moving_variance/readImlp_bot_layer/mlp_bot_hiddenlayer_3/batch_normalization/moments/Squeeze_1*
_output_shapes
:*
T0*Z
_classP
NLloc:@mlp_bot_layer/mlp_bot_hiddenlayer_3/batch_normalization/moving_variance
х
Mmlp_bot_layer/mlp_bot_hiddenlayer_3/batch_normalization/AssignMovingAvg_1/mulMulMmlp_bot_layer/mlp_bot_hiddenlayer_3/batch_normalization/AssignMovingAvg_1/subOmlp_bot_layer/mlp_bot_hiddenlayer_3/batch_normalization/AssignMovingAvg_1/decay*
T0*
_output_shapes
:*Z
_classP
NLloc:@mlp_bot_layer/mlp_bot_hiddenlayer_3/batch_normalization/moving_variance
п
Imlp_bot_layer/mlp_bot_hiddenlayer_3/batch_normalization/AssignMovingAvg_1	AssignSubGmlp_bot_layer/mlp_bot_hiddenlayer_3/batch_normalization/moving_varianceMmlp_bot_layer/mlp_bot_hiddenlayer_3/batch_normalization/AssignMovingAvg_1/mul*Z
_classP
NLloc:@mlp_bot_layer/mlp_bot_hiddenlayer_3/batch_normalization/moving_variance*
_output_shapes
:*
T0
Ј
<mlp_bot_layer/mlp_bot_hiddenlayer_3/batch_normalization/betaIdentityHmlp_bot_layer/mlp_bot_hiddenlayer_3/batch_normalization/beta/part_0/read*
_output_shapes
:*
T0
є
=mlp_bot_layer/mlp_bot_hiddenlayer_3/batch_normalization/gammaIdentityImlp_bot_layer/mlp_bot_hiddenlayer_3/batch_normalization/gamma/part_0/read*
T0*
_output_shapes
:
М
Gmlp_bot_layer/mlp_bot_hiddenlayer_3/batch_normalization/batchnorm/add/yConst*
valueB
 *oГ:*
dtype0*
_output_shapes
: 
З
Emlp_bot_layer/mlp_bot_hiddenlayer_3/batch_normalization/batchnorm/addAddV2Imlp_bot_layer/mlp_bot_hiddenlayer_3/batch_normalization/moments/Squeeze_1Gmlp_bot_layer/mlp_bot_hiddenlayer_3/batch_normalization/batchnorm/add/y*
T0*
_output_shapes
:
Љ
Gmlp_bot_layer/mlp_bot_hiddenlayer_3/batch_normalization/batchnorm/RsqrtRsqrtEmlp_bot_layer/mlp_bot_hiddenlayer_3/batch_normalization/batchnorm/add*
T0*
_output_shapes
:
щ
Emlp_bot_layer/mlp_bot_hiddenlayer_3/batch_normalization/batchnorm/mulMulGmlp_bot_layer/mlp_bot_hiddenlayer_3/batch_normalization/batchnorm/Rsqrt=mlp_bot_layer/mlp_bot_hiddenlayer_3/batch_normalization/gamma*
_output_shapes
:*
T0
с
Gmlp_bot_layer/mlp_bot_hiddenlayer_3/batch_normalization/batchnorm/mul_1Mul(mlp_bot_layer/mlp_bot_hiddenlayer_3/ReluEmlp_bot_layer/mlp_bot_hiddenlayer_3/batch_normalization/batchnorm/mul*'
_output_shapes
:€€€€€€€€€*
T0
Г
Gmlp_bot_layer/mlp_bot_hiddenlayer_3/batch_normalization/batchnorm/mul_2MulGmlp_bot_layer/mlp_bot_hiddenlayer_3/batch_normalization/moments/SqueezeEmlp_bot_layer/mlp_bot_hiddenlayer_3/batch_normalization/batchnorm/mul*
T0*
_output_shapes
:
ш
Emlp_bot_layer/mlp_bot_hiddenlayer_3/batch_normalization/batchnorm/subSub<mlp_bot_layer/mlp_bot_hiddenlayer_3/batch_normalization/betaGmlp_bot_layer/mlp_bot_hiddenlayer_3/batch_normalization/batchnorm/mul_2*
_output_shapes
:*
T0
Т
Gmlp_bot_layer/mlp_bot_hiddenlayer_3/batch_normalization/batchnorm/add_1AddV2Gmlp_bot_layer/mlp_bot_hiddenlayer_3/batch_normalization/batchnorm/mul_1Emlp_bot_layer/mlp_bot_hiddenlayer_3/batch_normalization/batchnorm/sub*'
_output_shapes
:€€€€€€€€€*
T0
Є
6mlp_bot_layer/mlp_bot_hiddenlayer_3/zero_fraction/SizeSizeGmlp_bot_layer/mlp_bot_hiddenlayer_3/batch_normalization/batchnorm/add_1*
out_type0	*
_output_shapes
: *
T0
Г
=mlp_bot_layer/mlp_bot_hiddenlayer_3/zero_fraction/LessEqual/yConst*
_output_shapes
: *
dtype0	*
valueB	 R€€€€
а
;mlp_bot_layer/mlp_bot_hiddenlayer_3/zero_fraction/LessEqual	LessEqual6mlp_bot_layer/mlp_bot_hiddenlayer_3/zero_fraction/Size=mlp_bot_layer/mlp_bot_hiddenlayer_3/zero_fraction/LessEqual/y*
_output_shapes
: *
T0	
д
=mlp_bot_layer/mlp_bot_hiddenlayer_3/zero_fraction/cond/SwitchSwitch;mlp_bot_layer/mlp_bot_hiddenlayer_3/zero_fraction/LessEqual;mlp_bot_layer/mlp_bot_hiddenlayer_3/zero_fraction/LessEqual*
_output_shapes
: : *
T0

≠
?mlp_bot_layer/mlp_bot_hiddenlayer_3/zero_fraction/cond/switch_tIdentity?mlp_bot_layer/mlp_bot_hiddenlayer_3/zero_fraction/cond/Switch:1*
_output_shapes
: *
T0

Ђ
?mlp_bot_layer/mlp_bot_hiddenlayer_3/zero_fraction/cond/switch_fIdentity=mlp_bot_layer/mlp_bot_hiddenlayer_3/zero_fraction/cond/Switch*
T0
*
_output_shapes
: 
®
>mlp_bot_layer/mlp_bot_hiddenlayer_3/zero_fraction/cond/pred_idIdentity;mlp_bot_layer/mlp_bot_hiddenlayer_3/zero_fraction/LessEqual*
T0
*
_output_shapes
: 
—
Jmlp_bot_layer/mlp_bot_hiddenlayer_3/zero_fraction/cond/count_nonzero/zerosConst@^mlp_bot_layer/mlp_bot_hiddenlayer_3/zero_fraction/cond/switch_t*
valueB
 *    *
_output_shapes
: *
dtype0
ѓ
Mmlp_bot_layer/mlp_bot_hiddenlayer_3/zero_fraction/cond/count_nonzero/NotEqualNotEqualVmlp_bot_layer/mlp_bot_hiddenlayer_3/zero_fraction/cond/count_nonzero/NotEqual/Switch:1Jmlp_bot_layer/mlp_bot_hiddenlayer_3/zero_fraction/cond/count_nonzero/zeros*
T0*'
_output_shapes
:€€€€€€€€€
И
Tmlp_bot_layer/mlp_bot_hiddenlayer_3/zero_fraction/cond/count_nonzero/NotEqual/SwitchSwitchGmlp_bot_layer/mlp_bot_hiddenlayer_3/batch_normalization/batchnorm/add_1>mlp_bot_layer/mlp_bot_hiddenlayer_3/zero_fraction/cond/pred_id*
T0*:
_output_shapes(
&:€€€€€€€€€:€€€€€€€€€*Z
_classP
NLloc:@mlp_bot_layer/mlp_bot_hiddenlayer_3/batch_normalization/batchnorm/add_1
б
Imlp_bot_layer/mlp_bot_hiddenlayer_3/zero_fraction/cond/count_nonzero/CastCastMmlp_bot_layer/mlp_bot_hiddenlayer_3/zero_fraction/cond/count_nonzero/NotEqual*'
_output_shapes
:€€€€€€€€€*

DstT0*

SrcT0

Ё
Jmlp_bot_layer/mlp_bot_hiddenlayer_3/zero_fraction/cond/count_nonzero/ConstConst@^mlp_bot_layer/mlp_bot_hiddenlayer_3/zero_fraction/cond/switch_t*
dtype0*
_output_shapes
:*
valueB"       
С
Rmlp_bot_layer/mlp_bot_hiddenlayer_3/zero_fraction/cond/count_nonzero/nonzero_countSumImlp_bot_layer/mlp_bot_hiddenlayer_3/zero_fraction/cond/count_nonzero/CastJmlp_bot_layer/mlp_bot_hiddenlayer_3/zero_fraction/cond/count_nonzero/Const*
T0*
_output_shapes
: 
«
;mlp_bot_layer/mlp_bot_hiddenlayer_3/zero_fraction/cond/CastCastRmlp_bot_layer/mlp_bot_hiddenlayer_3/zero_fraction/cond/count_nonzero/nonzero_count*

DstT0	*
_output_shapes
: *

SrcT0
”
Lmlp_bot_layer/mlp_bot_hiddenlayer_3/zero_fraction/cond/count_nonzero_1/zerosConst@^mlp_bot_layer/mlp_bot_hiddenlayer_3/zero_fraction/cond/switch_f*
dtype0*
_output_shapes
: *
valueB
 *    
≥
Omlp_bot_layer/mlp_bot_hiddenlayer_3/zero_fraction/cond/count_nonzero_1/NotEqualNotEqualVmlp_bot_layer/mlp_bot_hiddenlayer_3/zero_fraction/cond/count_nonzero_1/NotEqual/SwitchLmlp_bot_layer/mlp_bot_hiddenlayer_3/zero_fraction/cond/count_nonzero_1/zeros*'
_output_shapes
:€€€€€€€€€*
T0
К
Vmlp_bot_layer/mlp_bot_hiddenlayer_3/zero_fraction/cond/count_nonzero_1/NotEqual/SwitchSwitchGmlp_bot_layer/mlp_bot_hiddenlayer_3/batch_normalization/batchnorm/add_1>mlp_bot_layer/mlp_bot_hiddenlayer_3/zero_fraction/cond/pred_id*Z
_classP
NLloc:@mlp_bot_layer/mlp_bot_hiddenlayer_3/batch_normalization/batchnorm/add_1*
T0*:
_output_shapes(
&:€€€€€€€€€:€€€€€€€€€
е
Kmlp_bot_layer/mlp_bot_hiddenlayer_3/zero_fraction/cond/count_nonzero_1/CastCastOmlp_bot_layer/mlp_bot_hiddenlayer_3/zero_fraction/cond/count_nonzero_1/NotEqual*

SrcT0
*

DstT0	*'
_output_shapes
:€€€€€€€€€
я
Lmlp_bot_layer/mlp_bot_hiddenlayer_3/zero_fraction/cond/count_nonzero_1/ConstConst@^mlp_bot_layer/mlp_bot_hiddenlayer_3/zero_fraction/cond/switch_f*
_output_shapes
:*
dtype0*
valueB"       
Ч
Tmlp_bot_layer/mlp_bot_hiddenlayer_3/zero_fraction/cond/count_nonzero_1/nonzero_countSumKmlp_bot_layer/mlp_bot_hiddenlayer_3/zero_fraction/cond/count_nonzero_1/CastLmlp_bot_layer/mlp_bot_hiddenlayer_3/zero_fraction/cond/count_nonzero_1/Const*
T0	*
_output_shapes
: 
Д
<mlp_bot_layer/mlp_bot_hiddenlayer_3/zero_fraction/cond/MergeMergeTmlp_bot_layer/mlp_bot_hiddenlayer_3/zero_fraction/cond/count_nonzero_1/nonzero_count;mlp_bot_layer/mlp_bot_hiddenlayer_3/zero_fraction/cond/Cast*
N*
T0	*
_output_shapes
: : 
ж
Hmlp_bot_layer/mlp_bot_hiddenlayer_3/zero_fraction/counts_to_fraction/subSub6mlp_bot_layer/mlp_bot_hiddenlayer_3/zero_fraction/Size<mlp_bot_layer/mlp_bot_hiddenlayer_3/zero_fraction/cond/Merge*
T0	*
_output_shapes
: 
Ћ
Imlp_bot_layer/mlp_bot_hiddenlayer_3/zero_fraction/counts_to_fraction/CastCastHmlp_bot_layer/mlp_bot_hiddenlayer_3/zero_fraction/counts_to_fraction/sub*

DstT0*

SrcT0	*
_output_shapes
: 
ї
Kmlp_bot_layer/mlp_bot_hiddenlayer_3/zero_fraction/counts_to_fraction/Cast_1Cast6mlp_bot_layer/mlp_bot_hiddenlayer_3/zero_fraction/Size*
_output_shapes
: *

SrcT0	*

DstT0
Р
Lmlp_bot_layer/mlp_bot_hiddenlayer_3/zero_fraction/counts_to_fraction/truedivRealDivImlp_bot_layer/mlp_bot_hiddenlayer_3/zero_fraction/counts_to_fraction/CastKmlp_bot_layer/mlp_bot_hiddenlayer_3/zero_fraction/counts_to_fraction/Cast_1*
_output_shapes
: *
T0
µ
:mlp_bot_layer/mlp_bot_hiddenlayer_3/zero_fraction/fractionIdentityLmlp_bot_layer/mlp_bot_hiddenlayer_3/zero_fraction/counts_to_fraction/truediv*
T0*
_output_shapes
: 
Д
dmlp_bot_layer/mlp_bot_hiddenlayer_3/mlp_bot_layer/mlp_bot_hiddenlayer_3/fraction_of_zero_values/tagsConst*
_output_shapes
: *p
valuegBe B_mlp_bot_layer/mlp_bot_hiddenlayer_3/mlp_bot_layer/mlp_bot_hiddenlayer_3/fraction_of_zero_values*
dtype0
≥
_mlp_bot_layer/mlp_bot_hiddenlayer_3/mlp_bot_layer/mlp_bot_hiddenlayer_3/fraction_of_zero_valuesScalarSummarydmlp_bot_layer/mlp_bot_hiddenlayer_3/mlp_bot_layer/mlp_bot_hiddenlayer_3/fraction_of_zero_values/tags:mlp_bot_layer/mlp_bot_hiddenlayer_3/zero_fraction/fraction*
_output_shapes
: *
T0
й
Vmlp_bot_layer/mlp_bot_hiddenlayer_3/mlp_bot_layer/mlp_bot_hiddenlayer_3/activation/tagConst*
dtype0*
_output_shapes
: *c
valueZBX BRmlp_bot_layer/mlp_bot_hiddenlayer_3/mlp_bot_layer/mlp_bot_hiddenlayer_3/activation
Я
Rmlp_bot_layer/mlp_bot_hiddenlayer_3/mlp_bot_layer/mlp_bot_hiddenlayer_3/activationHistogramSummaryVmlp_bot_layer/mlp_bot_hiddenlayer_3/mlp_bot_layer/mlp_bot_hiddenlayer_3/activation/tagGmlp_bot_layer/mlp_bot_hiddenlayer_3/batch_normalization/batchnorm/add_1*
_output_shapes
: 
M
concat/axisConst*
value	B :*
dtype0*
_output_shapes
: 
„
concatConcatV2Gmlp_bot_layer/mlp_bot_hiddenlayer_3/batch_normalization/batchnorm/add_11input_layer/sparse_input_layer/input_layer/concatconcat/axis*
N*
T0*(
_output_shapes
:€€€€€€€€€∞
й
Rmlp_top_layer/mlp_top_hiddenlayer_0/kernel/part_0/Initializer/random_uniform/shapeConst*
_output_shapes
:*
dtype0*
valueB"∞     *D
_class:
86loc:@mlp_top_layer/mlp_top_hiddenlayer_0/kernel/part_0
џ
Pmlp_top_layer/mlp_top_hiddenlayer_0/kernel/part_0/Initializer/random_uniform/minConst*
_output_shapes
: *
dtype0*D
_class:
86loc:@mlp_top_layer/mlp_top_hiddenlayer_0/kernel/part_0*
valueB
 *dF£љ
џ
Pmlp_top_layer/mlp_top_hiddenlayer_0/kernel/part_0/Initializer/random_uniform/maxConst*
valueB
 *dF£=*
_output_shapes
: *D
_class:
86loc:@mlp_top_layer/mlp_top_hiddenlayer_0/kernel/part_0*
dtype0
љ
Zmlp_top_layer/mlp_top_hiddenlayer_0/kernel/part_0/Initializer/random_uniform/RandomUniformRandomUniformRmlp_top_layer/mlp_top_hiddenlayer_0/kernel/part_0/Initializer/random_uniform/shape*
T0*D
_class:
86loc:@mlp_top_layer/mlp_top_hiddenlayer_0/kernel/part_0*
dtype0* 
_output_shapes
:
∞А
в
Pmlp_top_layer/mlp_top_hiddenlayer_0/kernel/part_0/Initializer/random_uniform/subSubPmlp_top_layer/mlp_top_hiddenlayer_0/kernel/part_0/Initializer/random_uniform/maxPmlp_top_layer/mlp_top_hiddenlayer_0/kernel/part_0/Initializer/random_uniform/min*D
_class:
86loc:@mlp_top_layer/mlp_top_hiddenlayer_0/kernel/part_0*
T0*
_output_shapes
: 
ц
Pmlp_top_layer/mlp_top_hiddenlayer_0/kernel/part_0/Initializer/random_uniform/mulMulZmlp_top_layer/mlp_top_hiddenlayer_0/kernel/part_0/Initializer/random_uniform/RandomUniformPmlp_top_layer/mlp_top_hiddenlayer_0/kernel/part_0/Initializer/random_uniform/sub* 
_output_shapes
:
∞А*
T0*D
_class:
86loc:@mlp_top_layer/mlp_top_hiddenlayer_0/kernel/part_0
и
Lmlp_top_layer/mlp_top_hiddenlayer_0/kernel/part_0/Initializer/random_uniformAddPmlp_top_layer/mlp_top_hiddenlayer_0/kernel/part_0/Initializer/random_uniform/mulPmlp_top_layer/mlp_top_hiddenlayer_0/kernel/part_0/Initializer/random_uniform/min* 
_output_shapes
:
∞А*
T0*D
_class:
86loc:@mlp_top_layer/mlp_top_hiddenlayer_0/kernel/part_0
Ћ
1mlp_top_layer/mlp_top_hiddenlayer_0/kernel/part_0
VariableV2*
dtype0*
shape:
∞А*D
_class:
86loc:@mlp_top_layer/mlp_top_hiddenlayer_0/kernel/part_0* 
_output_shapes
:
∞А
і
8mlp_top_layer/mlp_top_hiddenlayer_0/kernel/part_0/AssignAssign1mlp_top_layer/mlp_top_hiddenlayer_0/kernel/part_0Lmlp_top_layer/mlp_top_hiddenlayer_0/kernel/part_0/Initializer/random_uniform*D
_class:
86loc:@mlp_top_layer/mlp_top_hiddenlayer_0/kernel/part_0* 
_output_shapes
:
∞А*
T0
ж
6mlp_top_layer/mlp_top_hiddenlayer_0/kernel/part_0/readIdentity1mlp_top_layer/mlp_top_hiddenlayer_0/kernel/part_0*D
_class:
86loc:@mlp_top_layer/mlp_top_hiddenlayer_0/kernel/part_0* 
_output_shapes
:
∞А*
T0
‘
Amlp_top_layer/mlp_top_hiddenlayer_0/bias/part_0/Initializer/zerosConst*
valueBА*    *
_output_shapes	
:А*
dtype0*B
_class8
64loc:@mlp_top_layer/mlp_top_hiddenlayer_0/bias/part_0
љ
/mlp_top_layer/mlp_top_hiddenlayer_0/bias/part_0
VariableV2*
dtype0*B
_class8
64loc:@mlp_top_layer/mlp_top_hiddenlayer_0/bias/part_0*
_output_shapes	
:А*
shape:А
Ю
6mlp_top_layer/mlp_top_hiddenlayer_0/bias/part_0/AssignAssign/mlp_top_layer/mlp_top_hiddenlayer_0/bias/part_0Amlp_top_layer/mlp_top_hiddenlayer_0/bias/part_0/Initializer/zeros*
T0*
_output_shapes	
:А*B
_class8
64loc:@mlp_top_layer/mlp_top_hiddenlayer_0/bias/part_0
џ
4mlp_top_layer/mlp_top_hiddenlayer_0/bias/part_0/readIdentity/mlp_top_layer/mlp_top_hiddenlayer_0/bias/part_0*B
_class8
64loc:@mlp_top_layer/mlp_top_hiddenlayer_0/bias/part_0*
T0*
_output_shapes	
:А
Щ
*mlp_top_layer/mlp_top_hiddenlayer_0/kernelIdentity6mlp_top_layer/mlp_top_hiddenlayer_0/kernel/part_0/read* 
_output_shapes
:
∞А*
T0
Ы
*mlp_top_layer/mlp_top_hiddenlayer_0/MatMulMatMulconcat*mlp_top_layer/mlp_top_hiddenlayer_0/kernel*(
_output_shapes
:€€€€€€€€€А*
T0
Р
(mlp_top_layer/mlp_top_hiddenlayer_0/biasIdentity4mlp_top_layer/mlp_top_hiddenlayer_0/bias/part_0/read*
_output_shapes	
:А*
T0
њ
+mlp_top_layer/mlp_top_hiddenlayer_0/BiasAddBiasAdd*mlp_top_layer/mlp_top_hiddenlayer_0/MatMul(mlp_top_layer/mlp_top_hiddenlayer_0/bias*
T0*(
_output_shapes
:€€€€€€€€€А
Р
(mlp_top_layer/mlp_top_hiddenlayer_0/ReluRelu+mlp_top_layer/mlp_top_hiddenlayer_0/BiasAdd*
T0*(
_output_shapes
:€€€€€€€€€А
Г
 mlp_top_layer/zero_fraction/SizeSize(mlp_top_layer/mlp_top_hiddenlayer_0/Relu*
T0*
out_type0	*
_output_shapes
: 
m
'mlp_top_layer/zero_fraction/LessEqual/yConst*
valueB	 R€€€€*
_output_shapes
: *
dtype0	
Ю
%mlp_top_layer/zero_fraction/LessEqual	LessEqual mlp_top_layer/zero_fraction/Size'mlp_top_layer/zero_fraction/LessEqual/y*
_output_shapes
: *
T0	
Ґ
'mlp_top_layer/zero_fraction/cond/SwitchSwitch%mlp_top_layer/zero_fraction/LessEqual%mlp_top_layer/zero_fraction/LessEqual*
_output_shapes
: : *
T0

Б
)mlp_top_layer/zero_fraction/cond/switch_tIdentity)mlp_top_layer/zero_fraction/cond/Switch:1*
_output_shapes
: *
T0


)mlp_top_layer/zero_fraction/cond/switch_fIdentity'mlp_top_layer/zero_fraction/cond/Switch*
_output_shapes
: *
T0

|
(mlp_top_layer/zero_fraction/cond/pred_idIdentity%mlp_top_layer/zero_fraction/LessEqual*
T0
*
_output_shapes
: 
•
4mlp_top_layer/zero_fraction/cond/count_nonzero/zerosConst*^mlp_top_layer/zero_fraction/cond/switch_t*
valueB
 *    *
_output_shapes
: *
dtype0
о
7mlp_top_layer/zero_fraction/cond/count_nonzero/NotEqualNotEqual@mlp_top_layer/zero_fraction/cond/count_nonzero/NotEqual/Switch:14mlp_top_layer/zero_fraction/cond/count_nonzero/zeros*
T0*(
_output_shapes
:€€€€€€€€€А
†
>mlp_top_layer/zero_fraction/cond/count_nonzero/NotEqual/SwitchSwitch(mlp_top_layer/mlp_top_hiddenlayer_0/Relu(mlp_top_layer/zero_fraction/cond/pred_id*<
_output_shapes*
(:€€€€€€€€€А:€€€€€€€€€А*;
_class1
/-loc:@mlp_top_layer/mlp_top_hiddenlayer_0/Relu*
T0
ґ
3mlp_top_layer/zero_fraction/cond/count_nonzero/CastCast7mlp_top_layer/zero_fraction/cond/count_nonzero/NotEqual*

DstT0*(
_output_shapes
:€€€€€€€€€А*

SrcT0

±
4mlp_top_layer/zero_fraction/cond/count_nonzero/ConstConst*^mlp_top_layer/zero_fraction/cond/switch_t*
dtype0*
_output_shapes
:*
valueB"       
ѕ
<mlp_top_layer/zero_fraction/cond/count_nonzero/nonzero_countSum3mlp_top_layer/zero_fraction/cond/count_nonzero/Cast4mlp_top_layer/zero_fraction/cond/count_nonzero/Const*
T0*
_output_shapes
: 
Ы
%mlp_top_layer/zero_fraction/cond/CastCast<mlp_top_layer/zero_fraction/cond/count_nonzero/nonzero_count*

SrcT0*

DstT0	*
_output_shapes
: 
І
6mlp_top_layer/zero_fraction/cond/count_nonzero_1/zerosConst*^mlp_top_layer/zero_fraction/cond/switch_f*
_output_shapes
: *
valueB
 *    *
dtype0
т
9mlp_top_layer/zero_fraction/cond/count_nonzero_1/NotEqualNotEqual@mlp_top_layer/zero_fraction/cond/count_nonzero_1/NotEqual/Switch6mlp_top_layer/zero_fraction/cond/count_nonzero_1/zeros*(
_output_shapes
:€€€€€€€€€А*
T0
Ґ
@mlp_top_layer/zero_fraction/cond/count_nonzero_1/NotEqual/SwitchSwitch(mlp_top_layer/mlp_top_hiddenlayer_0/Relu(mlp_top_layer/zero_fraction/cond/pred_id*;
_class1
/-loc:@mlp_top_layer/mlp_top_hiddenlayer_0/Relu*<
_output_shapes*
(:€€€€€€€€€А:€€€€€€€€€А*
T0
Ї
5mlp_top_layer/zero_fraction/cond/count_nonzero_1/CastCast9mlp_top_layer/zero_fraction/cond/count_nonzero_1/NotEqual*

SrcT0
*(
_output_shapes
:€€€€€€€€€А*

DstT0	
≥
6mlp_top_layer/zero_fraction/cond/count_nonzero_1/ConstConst*^mlp_top_layer/zero_fraction/cond/switch_f*
valueB"       *
dtype0*
_output_shapes
:
’
>mlp_top_layer/zero_fraction/cond/count_nonzero_1/nonzero_countSum5mlp_top_layer/zero_fraction/cond/count_nonzero_1/Cast6mlp_top_layer/zero_fraction/cond/count_nonzero_1/Const*
_output_shapes
: *
T0	
¬
&mlp_top_layer/zero_fraction/cond/MergeMerge>mlp_top_layer/zero_fraction/cond/count_nonzero_1/nonzero_count%mlp_top_layer/zero_fraction/cond/Cast*
T0	*
N*
_output_shapes
: : 
§
2mlp_top_layer/zero_fraction/counts_to_fraction/subSub mlp_top_layer/zero_fraction/Size&mlp_top_layer/zero_fraction/cond/Merge*
T0	*
_output_shapes
: 
Я
3mlp_top_layer/zero_fraction/counts_to_fraction/CastCast2mlp_top_layer/zero_fraction/counts_to_fraction/sub*
_output_shapes
: *

DstT0*

SrcT0	
П
5mlp_top_layer/zero_fraction/counts_to_fraction/Cast_1Cast mlp_top_layer/zero_fraction/Size*

DstT0*
_output_shapes
: *

SrcT0	
ќ
6mlp_top_layer/zero_fraction/counts_to_fraction/truedivRealDiv3mlp_top_layer/zero_fraction/counts_to_fraction/Cast5mlp_top_layer/zero_fraction/counts_to_fraction/Cast_1*
T0*
_output_shapes
: 
Й
$mlp_top_layer/zero_fraction/fractionIdentity6mlp_top_layer/zero_fraction/counts_to_fraction/truediv*
_output_shapes
: *
T0
Ў
Nmlp_top_layer/mlp_top_layer/mlp_top_hiddenlayer_0/fraction_of_zero_values/tagsConst*
_output_shapes
: *Z
valueQBO BImlp_top_layer/mlp_top_layer/mlp_top_hiddenlayer_0/fraction_of_zero_values*
dtype0
с
Imlp_top_layer/mlp_top_layer/mlp_top_hiddenlayer_0/fraction_of_zero_valuesScalarSummaryNmlp_top_layer/mlp_top_layer/mlp_top_hiddenlayer_0/fraction_of_zero_values/tags$mlp_top_layer/zero_fraction/fraction*
T0*
_output_shapes
: 
љ
@mlp_top_layer/mlp_top_layer/mlp_top_hiddenlayer_0/activation/tagConst*
dtype0*M
valueDBB B<mlp_top_layer/mlp_top_layer/mlp_top_hiddenlayer_0/activation*
_output_shapes
: 
‘
<mlp_top_layer/mlp_top_layer/mlp_top_hiddenlayer_0/activationHistogramSummary@mlp_top_layer/mlp_top_layer/mlp_top_hiddenlayer_0/activation/tag(mlp_top_layer/mlp_top_hiddenlayer_0/Relu*
_output_shapes
: 
й
Rmlp_top_layer/mlp_top_hiddenlayer_1/kernel/part_0/Initializer/random_uniform/shapeConst*D
_class:
86loc:@mlp_top_layer/mlp_top_hiddenlayer_1/kernel/part_0*
_output_shapes
:*
valueB"∞     *
dtype0
џ
Pmlp_top_layer/mlp_top_hiddenlayer_1/kernel/part_0/Initializer/random_uniform/minConst*
valueB
 *Aњљ*D
_class:
86loc:@mlp_top_layer/mlp_top_hiddenlayer_1/kernel/part_0*
_output_shapes
: *
dtype0
џ
Pmlp_top_layer/mlp_top_hiddenlayer_1/kernel/part_0/Initializer/random_uniform/maxConst*D
_class:
86loc:@mlp_top_layer/mlp_top_hiddenlayer_1/kernel/part_0*
dtype0*
_output_shapes
: *
valueB
 *Aњ=
љ
Zmlp_top_layer/mlp_top_hiddenlayer_1/kernel/part_0/Initializer/random_uniform/RandomUniformRandomUniformRmlp_top_layer/mlp_top_hiddenlayer_1/kernel/part_0/Initializer/random_uniform/shape*D
_class:
86loc:@mlp_top_layer/mlp_top_hiddenlayer_1/kernel/part_0*
dtype0* 
_output_shapes
:
∞А*
T0
в
Pmlp_top_layer/mlp_top_hiddenlayer_1/kernel/part_0/Initializer/random_uniform/subSubPmlp_top_layer/mlp_top_hiddenlayer_1/kernel/part_0/Initializer/random_uniform/maxPmlp_top_layer/mlp_top_hiddenlayer_1/kernel/part_0/Initializer/random_uniform/min*
T0*D
_class:
86loc:@mlp_top_layer/mlp_top_hiddenlayer_1/kernel/part_0*
_output_shapes
: 
ц
Pmlp_top_layer/mlp_top_hiddenlayer_1/kernel/part_0/Initializer/random_uniform/mulMulZmlp_top_layer/mlp_top_hiddenlayer_1/kernel/part_0/Initializer/random_uniform/RandomUniformPmlp_top_layer/mlp_top_hiddenlayer_1/kernel/part_0/Initializer/random_uniform/sub*
T0* 
_output_shapes
:
∞А*D
_class:
86loc:@mlp_top_layer/mlp_top_hiddenlayer_1/kernel/part_0
и
Lmlp_top_layer/mlp_top_hiddenlayer_1/kernel/part_0/Initializer/random_uniformAddPmlp_top_layer/mlp_top_hiddenlayer_1/kernel/part_0/Initializer/random_uniform/mulPmlp_top_layer/mlp_top_hiddenlayer_1/kernel/part_0/Initializer/random_uniform/min*D
_class:
86loc:@mlp_top_layer/mlp_top_hiddenlayer_1/kernel/part_0*
T0* 
_output_shapes
:
∞А
Ћ
1mlp_top_layer/mlp_top_hiddenlayer_1/kernel/part_0
VariableV2* 
_output_shapes
:
∞А*D
_class:
86loc:@mlp_top_layer/mlp_top_hiddenlayer_1/kernel/part_0*
dtype0*
shape:
∞А
і
8mlp_top_layer/mlp_top_hiddenlayer_1/kernel/part_0/AssignAssign1mlp_top_layer/mlp_top_hiddenlayer_1/kernel/part_0Lmlp_top_layer/mlp_top_hiddenlayer_1/kernel/part_0/Initializer/random_uniform*D
_class:
86loc:@mlp_top_layer/mlp_top_hiddenlayer_1/kernel/part_0* 
_output_shapes
:
∞А*
T0
ж
6mlp_top_layer/mlp_top_hiddenlayer_1/kernel/part_0/readIdentity1mlp_top_layer/mlp_top_hiddenlayer_1/kernel/part_0* 
_output_shapes
:
∞А*D
_class:
86loc:@mlp_top_layer/mlp_top_hiddenlayer_1/kernel/part_0*
T0
‘
Amlp_top_layer/mlp_top_hiddenlayer_1/bias/part_0/Initializer/zerosConst*B
_class8
64loc:@mlp_top_layer/mlp_top_hiddenlayer_1/bias/part_0*
dtype0*
_output_shapes	
:А*
valueBА*    
љ
/mlp_top_layer/mlp_top_hiddenlayer_1/bias/part_0
VariableV2*
dtype0*B
_class8
64loc:@mlp_top_layer/mlp_top_hiddenlayer_1/bias/part_0*
_output_shapes	
:А*
shape:А
Ю
6mlp_top_layer/mlp_top_hiddenlayer_1/bias/part_0/AssignAssign/mlp_top_layer/mlp_top_hiddenlayer_1/bias/part_0Amlp_top_layer/mlp_top_hiddenlayer_1/bias/part_0/Initializer/zeros*
_output_shapes	
:А*
T0*B
_class8
64loc:@mlp_top_layer/mlp_top_hiddenlayer_1/bias/part_0
џ
4mlp_top_layer/mlp_top_hiddenlayer_1/bias/part_0/readIdentity/mlp_top_layer/mlp_top_hiddenlayer_1/bias/part_0*B
_class8
64loc:@mlp_top_layer/mlp_top_hiddenlayer_1/bias/part_0*
T0*
_output_shapes	
:А
Щ
*mlp_top_layer/mlp_top_hiddenlayer_1/kernelIdentity6mlp_top_layer/mlp_top_hiddenlayer_1/kernel/part_0/read* 
_output_shapes
:
∞А*
T0
Ы
*mlp_top_layer/mlp_top_hiddenlayer_1/MatMulMatMulconcat*mlp_top_layer/mlp_top_hiddenlayer_1/kernel*(
_output_shapes
:€€€€€€€€€А*
T0
Р
(mlp_top_layer/mlp_top_hiddenlayer_1/biasIdentity4mlp_top_layer/mlp_top_hiddenlayer_1/bias/part_0/read*
_output_shapes	
:А*
T0
њ
+mlp_top_layer/mlp_top_hiddenlayer_1/BiasAddBiasAdd*mlp_top_layer/mlp_top_hiddenlayer_1/MatMul(mlp_top_layer/mlp_top_hiddenlayer_1/bias*(
_output_shapes
:€€€€€€€€€А*
T0
Р
(mlp_top_layer/mlp_top_hiddenlayer_1/ReluRelu+mlp_top_layer/mlp_top_hiddenlayer_1/BiasAdd*
T0*(
_output_shapes
:€€€€€€€€€А
Е
"mlp_top_layer/zero_fraction_1/SizeSize(mlp_top_layer/mlp_top_hiddenlayer_1/Relu*
T0*
out_type0	*
_output_shapes
: 
o
)mlp_top_layer/zero_fraction_1/LessEqual/yConst*
dtype0	*
_output_shapes
: *
valueB	 R€€€€
§
'mlp_top_layer/zero_fraction_1/LessEqual	LessEqual"mlp_top_layer/zero_fraction_1/Size)mlp_top_layer/zero_fraction_1/LessEqual/y*
T0	*
_output_shapes
: 
®
)mlp_top_layer/zero_fraction_1/cond/SwitchSwitch'mlp_top_layer/zero_fraction_1/LessEqual'mlp_top_layer/zero_fraction_1/LessEqual*
T0
*
_output_shapes
: : 
Е
+mlp_top_layer/zero_fraction_1/cond/switch_tIdentity+mlp_top_layer/zero_fraction_1/cond/Switch:1*
T0
*
_output_shapes
: 
Г
+mlp_top_layer/zero_fraction_1/cond/switch_fIdentity)mlp_top_layer/zero_fraction_1/cond/Switch*
T0
*
_output_shapes
: 
А
*mlp_top_layer/zero_fraction_1/cond/pred_idIdentity'mlp_top_layer/zero_fraction_1/LessEqual*
_output_shapes
: *
T0

©
6mlp_top_layer/zero_fraction_1/cond/count_nonzero/zerosConst,^mlp_top_layer/zero_fraction_1/cond/switch_t*
dtype0*
valueB
 *    *
_output_shapes
: 
ф
9mlp_top_layer/zero_fraction_1/cond/count_nonzero/NotEqualNotEqualBmlp_top_layer/zero_fraction_1/cond/count_nonzero/NotEqual/Switch:16mlp_top_layer/zero_fraction_1/cond/count_nonzero/zeros*
T0*(
_output_shapes
:€€€€€€€€€А
§
@mlp_top_layer/zero_fraction_1/cond/count_nonzero/NotEqual/SwitchSwitch(mlp_top_layer/mlp_top_hiddenlayer_1/Relu*mlp_top_layer/zero_fraction_1/cond/pred_id*
T0*<
_output_shapes*
(:€€€€€€€€€А:€€€€€€€€€А*;
_class1
/-loc:@mlp_top_layer/mlp_top_hiddenlayer_1/Relu
Ї
5mlp_top_layer/zero_fraction_1/cond/count_nonzero/CastCast9mlp_top_layer/zero_fraction_1/cond/count_nonzero/NotEqual*

SrcT0
*(
_output_shapes
:€€€€€€€€€А*

DstT0
µ
6mlp_top_layer/zero_fraction_1/cond/count_nonzero/ConstConst,^mlp_top_layer/zero_fraction_1/cond/switch_t*
valueB"       *
dtype0*
_output_shapes
:
’
>mlp_top_layer/zero_fraction_1/cond/count_nonzero/nonzero_countSum5mlp_top_layer/zero_fraction_1/cond/count_nonzero/Cast6mlp_top_layer/zero_fraction_1/cond/count_nonzero/Const*
T0*
_output_shapes
: 
Я
'mlp_top_layer/zero_fraction_1/cond/CastCast>mlp_top_layer/zero_fraction_1/cond/count_nonzero/nonzero_count*

DstT0	*
_output_shapes
: *

SrcT0
Ђ
8mlp_top_layer/zero_fraction_1/cond/count_nonzero_1/zerosConst,^mlp_top_layer/zero_fraction_1/cond/switch_f*
valueB
 *    *
dtype0*
_output_shapes
: 
ш
;mlp_top_layer/zero_fraction_1/cond/count_nonzero_1/NotEqualNotEqualBmlp_top_layer/zero_fraction_1/cond/count_nonzero_1/NotEqual/Switch8mlp_top_layer/zero_fraction_1/cond/count_nonzero_1/zeros*
T0*(
_output_shapes
:€€€€€€€€€А
¶
Bmlp_top_layer/zero_fraction_1/cond/count_nonzero_1/NotEqual/SwitchSwitch(mlp_top_layer/mlp_top_hiddenlayer_1/Relu*mlp_top_layer/zero_fraction_1/cond/pred_id*
T0*;
_class1
/-loc:@mlp_top_layer/mlp_top_hiddenlayer_1/Relu*<
_output_shapes*
(:€€€€€€€€€А:€€€€€€€€€А
Њ
7mlp_top_layer/zero_fraction_1/cond/count_nonzero_1/CastCast;mlp_top_layer/zero_fraction_1/cond/count_nonzero_1/NotEqual*

SrcT0
*

DstT0	*(
_output_shapes
:€€€€€€€€€А
Ј
8mlp_top_layer/zero_fraction_1/cond/count_nonzero_1/ConstConst,^mlp_top_layer/zero_fraction_1/cond/switch_f*
_output_shapes
:*
valueB"       *
dtype0
џ
@mlp_top_layer/zero_fraction_1/cond/count_nonzero_1/nonzero_countSum7mlp_top_layer/zero_fraction_1/cond/count_nonzero_1/Cast8mlp_top_layer/zero_fraction_1/cond/count_nonzero_1/Const*
_output_shapes
: *
T0	
»
(mlp_top_layer/zero_fraction_1/cond/MergeMerge@mlp_top_layer/zero_fraction_1/cond/count_nonzero_1/nonzero_count'mlp_top_layer/zero_fraction_1/cond/Cast*
N*
T0	*
_output_shapes
: : 
™
4mlp_top_layer/zero_fraction_1/counts_to_fraction/subSub"mlp_top_layer/zero_fraction_1/Size(mlp_top_layer/zero_fraction_1/cond/Merge*
_output_shapes
: *
T0	
£
5mlp_top_layer/zero_fraction_1/counts_to_fraction/CastCast4mlp_top_layer/zero_fraction_1/counts_to_fraction/sub*

DstT0*

SrcT0	*
_output_shapes
: 
У
7mlp_top_layer/zero_fraction_1/counts_to_fraction/Cast_1Cast"mlp_top_layer/zero_fraction_1/Size*

SrcT0	*

DstT0*
_output_shapes
: 
‘
8mlp_top_layer/zero_fraction_1/counts_to_fraction/truedivRealDiv5mlp_top_layer/zero_fraction_1/counts_to_fraction/Cast7mlp_top_layer/zero_fraction_1/counts_to_fraction/Cast_1*
_output_shapes
: *
T0
Н
&mlp_top_layer/zero_fraction_1/fractionIdentity8mlp_top_layer/zero_fraction_1/counts_to_fraction/truediv*
_output_shapes
: *
T0
Ў
Nmlp_top_layer/mlp_top_layer/mlp_top_hiddenlayer_1/fraction_of_zero_values/tagsConst*
_output_shapes
: *
dtype0*Z
valueQBO BImlp_top_layer/mlp_top_layer/mlp_top_hiddenlayer_1/fraction_of_zero_values
у
Imlp_top_layer/mlp_top_layer/mlp_top_hiddenlayer_1/fraction_of_zero_valuesScalarSummaryNmlp_top_layer/mlp_top_layer/mlp_top_hiddenlayer_1/fraction_of_zero_values/tags&mlp_top_layer/zero_fraction_1/fraction*
T0*
_output_shapes
: 
љ
@mlp_top_layer/mlp_top_layer/mlp_top_hiddenlayer_1/activation/tagConst*
dtype0*M
valueDBB B<mlp_top_layer/mlp_top_layer/mlp_top_hiddenlayer_1/activation*
_output_shapes
: 
‘
<mlp_top_layer/mlp_top_layer/mlp_top_hiddenlayer_1/activationHistogramSummary@mlp_top_layer/mlp_top_layer/mlp_top_hiddenlayer_1/activation/tag(mlp_top_layer/mlp_top_hiddenlayer_1/Relu*
_output_shapes
: 
°
.logits/kernel/Initializer/random_uniform/shapeConst*
dtype0* 
_class
loc:@logits/kernel*
valueB"      *
_output_shapes
:
У
,logits/kernel/Initializer/random_uniform/minConst*
valueB
 *IvЊ*
dtype0* 
_class
loc:@logits/kernel*
_output_shapes
: 
У
,logits/kernel/Initializer/random_uniform/maxConst*
dtype0* 
_class
loc:@logits/kernel*
valueB
 *Iv>*
_output_shapes
: 
–
6logits/kernel/Initializer/random_uniform/RandomUniformRandomUniform.logits/kernel/Initializer/random_uniform/shape*
_output_shapes
:	А*
dtype0*
T0* 
_class
loc:@logits/kernel
“
,logits/kernel/Initializer/random_uniform/subSub,logits/kernel/Initializer/random_uniform/max,logits/kernel/Initializer/random_uniform/min*
_output_shapes
: * 
_class
loc:@logits/kernel*
T0
е
,logits/kernel/Initializer/random_uniform/mulMul6logits/kernel/Initializer/random_uniform/RandomUniform,logits/kernel/Initializer/random_uniform/sub* 
_class
loc:@logits/kernel*
_output_shapes
:	А*
T0
„
(logits/kernel/Initializer/random_uniformAdd,logits/kernel/Initializer/random_uniform/mul,logits/kernel/Initializer/random_uniform/min* 
_class
loc:@logits/kernel*
_output_shapes
:	А*
T0
Б
logits/kernel
VariableV2*
shape:	А*
_output_shapes
:	А* 
_class
loc:@logits/kernel*
dtype0
£
logits/kernel/AssignAssignlogits/kernel(logits/kernel/Initializer/random_uniform*
_output_shapes
:	А* 
_class
loc:@logits/kernel*
T0
y
logits/kernel/readIdentitylogits/kernel* 
_class
loc:@logits/kernel*
_output_shapes
:	А*
T0
К
logits/bias/Initializer/zerosConst*
_class
loc:@logits/bias*
_output_shapes
:*
dtype0*
valueB*    
s
logits/bias
VariableV2*
shape:*
dtype0*
_output_shapes
:*
_class
loc:@logits/bias
Н
logits/bias/AssignAssignlogits/biaslogits/bias/Initializer/zeros*
_class
loc:@logits/bias*
_output_shapes
:*
T0
n
logits/bias/readIdentitylogits/bias*
_output_shapes
:*
T0*
_class
loc:@logits/bias
З
logits/MatMulMatMul(mlp_top_layer/mlp_top_hiddenlayer_1/Relulogits/kernel/read*'
_output_shapes
:€€€€€€€€€*
T0
l
logits/BiasAddBiasAddlogits/MatMullogits/bias/read*'
_output_shapes
:€€€€€€€€€*
T0
[
logits/SigmoidSigmoidlogits/BiasAdd*'
_output_shapes
:€€€€€€€€€*
T0
W
logits/RoundRoundlogits/Sigmoid*
T0*'
_output_shapes
:€€€€€€€€€
b
logits/zero_fraction/SizeSizelogits/Sigmoid*
out_type0	*
T0*
_output_shapes
: 
f
 logits/zero_fraction/LessEqual/yConst*
valueB	 R€€€€*
dtype0	*
_output_shapes
: 
Й
logits/zero_fraction/LessEqual	LessEquallogits/zero_fraction/Size logits/zero_fraction/LessEqual/y*
_output_shapes
: *
T0	
Н
 logits/zero_fraction/cond/SwitchSwitchlogits/zero_fraction/LessEquallogits/zero_fraction/LessEqual*
_output_shapes
: : *
T0

s
"logits/zero_fraction/cond/switch_tIdentity"logits/zero_fraction/cond/Switch:1*
_output_shapes
: *
T0

q
"logits/zero_fraction/cond/switch_fIdentity logits/zero_fraction/cond/Switch*
T0
*
_output_shapes
: 
n
!logits/zero_fraction/cond/pred_idIdentitylogits/zero_fraction/LessEqual*
_output_shapes
: *
T0

Ч
-logits/zero_fraction/cond/count_nonzero/zerosConst#^logits/zero_fraction/cond/switch_t*
_output_shapes
: *
valueB
 *    *
dtype0
Ў
0logits/zero_fraction/cond/count_nonzero/NotEqualNotEqual9logits/zero_fraction/cond/count_nonzero/NotEqual/Switch:1-logits/zero_fraction/cond/count_nonzero/zeros*'
_output_shapes
:€€€€€€€€€*
T0
№
7logits/zero_fraction/cond/count_nonzero/NotEqual/SwitchSwitchlogits/Sigmoid!logits/zero_fraction/cond/pred_id*
T0*:
_output_shapes(
&:€€€€€€€€€:€€€€€€€€€*!
_class
loc:@logits/Sigmoid
І
,logits/zero_fraction/cond/count_nonzero/CastCast0logits/zero_fraction/cond/count_nonzero/NotEqual*

SrcT0
*'
_output_shapes
:€€€€€€€€€*

DstT0
£
-logits/zero_fraction/cond/count_nonzero/ConstConst#^logits/zero_fraction/cond/switch_t*
valueB"       *
_output_shapes
:*
dtype0
Ї
5logits/zero_fraction/cond/count_nonzero/nonzero_countSum,logits/zero_fraction/cond/count_nonzero/Cast-logits/zero_fraction/cond/count_nonzero/Const*
_output_shapes
: *
T0
Н
logits/zero_fraction/cond/CastCast5logits/zero_fraction/cond/count_nonzero/nonzero_count*

DstT0	*
_output_shapes
: *

SrcT0
Щ
/logits/zero_fraction/cond/count_nonzero_1/zerosConst#^logits/zero_fraction/cond/switch_f*
dtype0*
valueB
 *    *
_output_shapes
: 
№
2logits/zero_fraction/cond/count_nonzero_1/NotEqualNotEqual9logits/zero_fraction/cond/count_nonzero_1/NotEqual/Switch/logits/zero_fraction/cond/count_nonzero_1/zeros*'
_output_shapes
:€€€€€€€€€*
T0
ё
9logits/zero_fraction/cond/count_nonzero_1/NotEqual/SwitchSwitchlogits/Sigmoid!logits/zero_fraction/cond/pred_id*:
_output_shapes(
&:€€€€€€€€€:€€€€€€€€€*!
_class
loc:@logits/Sigmoid*
T0
Ђ
.logits/zero_fraction/cond/count_nonzero_1/CastCast2logits/zero_fraction/cond/count_nonzero_1/NotEqual*

SrcT0
*'
_output_shapes
:€€€€€€€€€*

DstT0	
•
/logits/zero_fraction/cond/count_nonzero_1/ConstConst#^logits/zero_fraction/cond/switch_f*
_output_shapes
:*
dtype0*
valueB"       
ј
7logits/zero_fraction/cond/count_nonzero_1/nonzero_countSum.logits/zero_fraction/cond/count_nonzero_1/Cast/logits/zero_fraction/cond/count_nonzero_1/Const*
T0	*
_output_shapes
: 
≠
logits/zero_fraction/cond/MergeMerge7logits/zero_fraction/cond/count_nonzero_1/nonzero_countlogits/zero_fraction/cond/Cast*
T0	*
N*
_output_shapes
: : 
П
+logits/zero_fraction/counts_to_fraction/subSublogits/zero_fraction/Sizelogits/zero_fraction/cond/Merge*
T0	*
_output_shapes
: 
С
,logits/zero_fraction/counts_to_fraction/CastCast+logits/zero_fraction/counts_to_fraction/sub*
_output_shapes
: *

SrcT0	*

DstT0
Б
.logits/zero_fraction/counts_to_fraction/Cast_1Castlogits/zero_fraction/Size*
_output_shapes
: *

DstT0*

SrcT0	
є
/logits/zero_fraction/counts_to_fraction/truedivRealDiv,logits/zero_fraction/counts_to_fraction/Cast.logits/zero_fraction/counts_to_fraction/Cast_1*
T0*
_output_shapes
: 
{
logits/zero_fraction/fractionIdentity/logits/zero_fraction/counts_to_fraction/truediv*
T0*
_output_shapes
: 
Р
*logits/logits/fraction_of_zero_values/tagsConst*6
value-B+ B%logits/logits/fraction_of_zero_values*
dtype0*
_output_shapes
: 
Ґ
%logits/logits/fraction_of_zero_valuesScalarSummary*logits/logits/fraction_of_zero_values/tagslogits/zero_fraction/fraction*
_output_shapes
: *
T0
u
logits/logits/activation/tagConst*
_output_shapes
: *
dtype0*)
value B Blogits/logits/activation
r
logits/logits/activationHistogramSummarylogits/logits/activation/taglogits/Sigmoid*
_output_shapes
: 

initNoOp

init_all_tablesNoOp

init_1NoOp
4

group_depsNoOp^init^init_1^init_all_tables
Y
save/filename/inputConst*
_output_shapes
: *
valueB Bmodel*
dtype0
n
save/filenamePlaceholderWithDefaultsave/filename/input*
_output_shapes
: *
dtype0*
shape: 
e

save/ConstPlaceholderWithDefaultsave/filename*
shape: *
dtype0*
_output_shapes
: 
Д
save/StringJoin/inputs_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+_temp_52f25700074246d8a282430657463717/part
d
save/StringJoin
StringJoin
save/Constsave/StringJoin/inputs_1*
N*
_output_shapes
: 
Q
save/num_shardsConst*
_output_shapes
: *
value	B :*
dtype0
k
save/ShardedFilename/shardConst"/device:CPU:0*
value	B : *
dtype0*
_output_shapes
: 
М
save/ShardedFilenameShardedFilenamesave/StringJoinsave/ShardedFilename/shardsave/num_shards"/device:CPU:0*
_output_shapes
: 
щ
save/SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:9*
dtype0*Э
valueУBР9Bglobal_stepBJinput_layer/sparse_input_layer/input_layer/C10_embedding/embedding_weightsBJinput_layer/sparse_input_layer/input_layer/C11_embedding/embedding_weightsBJinput_layer/sparse_input_layer/input_layer/C12_embedding/embedding_weightsBJinput_layer/sparse_input_layer/input_layer/C13_embedding/embedding_weightsBJinput_layer/sparse_input_layer/input_layer/C14_embedding/embedding_weightsBJinput_layer/sparse_input_layer/input_layer/C15_embedding/embedding_weightsBJinput_layer/sparse_input_layer/input_layer/C16_embedding/embedding_weightsBJinput_layer/sparse_input_layer/input_layer/C17_embedding/embedding_weightsBJinput_layer/sparse_input_layer/input_layer/C18_embedding/embedding_weightsBJinput_layer/sparse_input_layer/input_layer/C19_embedding/embedding_weightsBIinput_layer/sparse_input_layer/input_layer/C1_embedding/embedding_weightsBJinput_layer/sparse_input_layer/input_layer/C20_embedding/embedding_weightsBJinput_layer/sparse_input_layer/input_layer/C21_embedding/embedding_weightsBJinput_layer/sparse_input_layer/input_layer/C22_embedding/embedding_weightsBJinput_layer/sparse_input_layer/input_layer/C23_embedding/embedding_weightsBJinput_layer/sparse_input_layer/input_layer/C24_embedding/embedding_weightsBJinput_layer/sparse_input_layer/input_layer/C25_embedding/embedding_weightsBJinput_layer/sparse_input_layer/input_layer/C26_embedding/embedding_weightsBIinput_layer/sparse_input_layer/input_layer/C2_embedding/embedding_weightsBIinput_layer/sparse_input_layer/input_layer/C3_embedding/embedding_weightsBIinput_layer/sparse_input_layer/input_layer/C4_embedding/embedding_weightsBIinput_layer/sparse_input_layer/input_layer/C5_embedding/embedding_weightsBIinput_layer/sparse_input_layer/input_layer/C6_embedding/embedding_weightsBIinput_layer/sparse_input_layer/input_layer/C7_embedding/embedding_weightsBIinput_layer/sparse_input_layer/input_layer/C8_embedding/embedding_weightsBIinput_layer/sparse_input_layer/input_layer/C9_embedding/embedding_weightsBlogits/biasBlogits/kernelB<mlp_bot_layer/mlp_bot_hiddenlayer_0/batch_normalization/betaB=mlp_bot_layer/mlp_bot_hiddenlayer_0/batch_normalization/gammaBCmlp_bot_layer/mlp_bot_hiddenlayer_0/batch_normalization/moving_meanBGmlp_bot_layer/mlp_bot_hiddenlayer_0/batch_normalization/moving_varianceB(mlp_bot_layer/mlp_bot_hiddenlayer_0/biasB*mlp_bot_layer/mlp_bot_hiddenlayer_0/kernelB<mlp_bot_layer/mlp_bot_hiddenlayer_1/batch_normalization/betaB=mlp_bot_layer/mlp_bot_hiddenlayer_1/batch_normalization/gammaBCmlp_bot_layer/mlp_bot_hiddenlayer_1/batch_normalization/moving_meanBGmlp_bot_layer/mlp_bot_hiddenlayer_1/batch_normalization/moving_varianceB(mlp_bot_layer/mlp_bot_hiddenlayer_1/biasB*mlp_bot_layer/mlp_bot_hiddenlayer_1/kernelB<mlp_bot_layer/mlp_bot_hiddenlayer_2/batch_normalization/betaB=mlp_bot_layer/mlp_bot_hiddenlayer_2/batch_normalization/gammaBCmlp_bot_layer/mlp_bot_hiddenlayer_2/batch_normalization/moving_meanBGmlp_bot_layer/mlp_bot_hiddenlayer_2/batch_normalization/moving_varianceB(mlp_bot_layer/mlp_bot_hiddenlayer_2/biasB*mlp_bot_layer/mlp_bot_hiddenlayer_2/kernelB<mlp_bot_layer/mlp_bot_hiddenlayer_3/batch_normalization/betaB=mlp_bot_layer/mlp_bot_hiddenlayer_3/batch_normalization/gammaBCmlp_bot_layer/mlp_bot_hiddenlayer_3/batch_normalization/moving_meanBGmlp_bot_layer/mlp_bot_hiddenlayer_3/batch_normalization/moving_varianceB(mlp_bot_layer/mlp_bot_hiddenlayer_3/biasB*mlp_bot_layer/mlp_bot_hiddenlayer_3/kernelB(mlp_top_layer/mlp_top_hiddenlayer_0/biasB*mlp_top_layer/mlp_top_hiddenlayer_0/kernelB(mlp_top_layer/mlp_top_hiddenlayer_1/biasB*mlp_top_layer/mlp_top_hiddenlayer_1/kernel
√
save/SaveV2/shape_and_slicesConst"/device:CPU:0*г
valueўB÷9B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B	512 0,512B	512 0,512B B B	512 0,512B13 512 0,13:0,512B	256 0,256B	256 0,256B B B	256 0,256B512 256 0,512:0,256B64 0,64B64 0,64B B B64 0,64B256 64 0,256:0,64B16 0,16B16 0,16B B B16 0,16B64 16 0,64:0,16B	512 0,512B432 512 0,432:0,512B	256 0,256B432 256 0,432:0,256*
dtype0*
_output_shapes
:9
≥
save/SaveV2SaveV2save/ShardedFilenamesave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesglobal_stepJinput_layer/sparse_input_layer/input_layer/C10_embedding/embedding_weightsJinput_layer/sparse_input_layer/input_layer/C11_embedding/embedding_weightsJinput_layer/sparse_input_layer/input_layer/C12_embedding/embedding_weightsJinput_layer/sparse_input_layer/input_layer/C13_embedding/embedding_weightsJinput_layer/sparse_input_layer/input_layer/C14_embedding/embedding_weightsJinput_layer/sparse_input_layer/input_layer/C15_embedding/embedding_weightsJinput_layer/sparse_input_layer/input_layer/C16_embedding/embedding_weightsJinput_layer/sparse_input_layer/input_layer/C17_embedding/embedding_weightsJinput_layer/sparse_input_layer/input_layer/C18_embedding/embedding_weightsJinput_layer/sparse_input_layer/input_layer/C19_embedding/embedding_weightsIinput_layer/sparse_input_layer/input_layer/C1_embedding/embedding_weightsJinput_layer/sparse_input_layer/input_layer/C20_embedding/embedding_weightsJinput_layer/sparse_input_layer/input_layer/C21_embedding/embedding_weightsJinput_layer/sparse_input_layer/input_layer/C22_embedding/embedding_weightsJinput_layer/sparse_input_layer/input_layer/C23_embedding/embedding_weightsJinput_layer/sparse_input_layer/input_layer/C24_embedding/embedding_weightsJinput_layer/sparse_input_layer/input_layer/C25_embedding/embedding_weightsJinput_layer/sparse_input_layer/input_layer/C26_embedding/embedding_weightsIinput_layer/sparse_input_layer/input_layer/C2_embedding/embedding_weightsIinput_layer/sparse_input_layer/input_layer/C3_embedding/embedding_weightsIinput_layer/sparse_input_layer/input_layer/C4_embedding/embedding_weightsIinput_layer/sparse_input_layer/input_layer/C5_embedding/embedding_weightsIinput_layer/sparse_input_layer/input_layer/C6_embedding/embedding_weightsIinput_layer/sparse_input_layer/input_layer/C7_embedding/embedding_weightsIinput_layer/sparse_input_layer/input_layer/C8_embedding/embedding_weightsIinput_layer/sparse_input_layer/input_layer/C9_embedding/embedding_weightslogits/biaslogits/kernelHmlp_bot_layer/mlp_bot_hiddenlayer_0/batch_normalization/beta/part_0/readImlp_bot_layer/mlp_bot_hiddenlayer_0/batch_normalization/gamma/part_0/readCmlp_bot_layer/mlp_bot_hiddenlayer_0/batch_normalization/moving_meanGmlp_bot_layer/mlp_bot_hiddenlayer_0/batch_normalization/moving_variance4mlp_bot_layer/mlp_bot_hiddenlayer_0/bias/part_0/read6mlp_bot_layer/mlp_bot_hiddenlayer_0/kernel/part_0/readHmlp_bot_layer/mlp_bot_hiddenlayer_1/batch_normalization/beta/part_0/readImlp_bot_layer/mlp_bot_hiddenlayer_1/batch_normalization/gamma/part_0/readCmlp_bot_layer/mlp_bot_hiddenlayer_1/batch_normalization/moving_meanGmlp_bot_layer/mlp_bot_hiddenlayer_1/batch_normalization/moving_variance4mlp_bot_layer/mlp_bot_hiddenlayer_1/bias/part_0/read6mlp_bot_layer/mlp_bot_hiddenlayer_1/kernel/part_0/readHmlp_bot_layer/mlp_bot_hiddenlayer_2/batch_normalization/beta/part_0/readImlp_bot_layer/mlp_bot_hiddenlayer_2/batch_normalization/gamma/part_0/readCmlp_bot_layer/mlp_bot_hiddenlayer_2/batch_normalization/moving_meanGmlp_bot_layer/mlp_bot_hiddenlayer_2/batch_normalization/moving_variance4mlp_bot_layer/mlp_bot_hiddenlayer_2/bias/part_0/read6mlp_bot_layer/mlp_bot_hiddenlayer_2/kernel/part_0/readHmlp_bot_layer/mlp_bot_hiddenlayer_3/batch_normalization/beta/part_0/readImlp_bot_layer/mlp_bot_hiddenlayer_3/batch_normalization/gamma/part_0/readCmlp_bot_layer/mlp_bot_hiddenlayer_3/batch_normalization/moving_meanGmlp_bot_layer/mlp_bot_hiddenlayer_3/batch_normalization/moving_variance4mlp_bot_layer/mlp_bot_hiddenlayer_3/bias/part_0/read6mlp_bot_layer/mlp_bot_hiddenlayer_3/kernel/part_0/read4mlp_top_layer/mlp_top_hiddenlayer_0/bias/part_0/read6mlp_top_layer/mlp_top_hiddenlayer_0/kernel/part_0/read4mlp_top_layer/mlp_top_hiddenlayer_1/bias/part_0/read6mlp_top_layer/mlp_top_hiddenlayer_1/kernel/part_0/read"/device:CPU:0*G
dtypes=
;29	
†
save/control_dependencyIdentitysave/ShardedFilename^save/SaveV2"/device:CPU:0*
_output_shapes
: *'
_class
loc:@save/ShardedFilename*
T0
†
+save/MergeV2Checkpoints/checkpoint_prefixesPacksave/ShardedFilename^save/control_dependency"/device:CPU:0*
N*
T0*
_output_shapes
:
u
save/MergeV2CheckpointsMergeV2Checkpoints+save/MergeV2Checkpoints/checkpoint_prefixes
save/Const"/device:CPU:0
Й
save/IdentityIdentity
save/Const^save/MergeV2Checkpoints^save/control_dependency"/device:CPU:0*
T0*
_output_shapes
: 
ь
save/RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:9*
dtype0*Э
valueУBР9Bglobal_stepBJinput_layer/sparse_input_layer/input_layer/C10_embedding/embedding_weightsBJinput_layer/sparse_input_layer/input_layer/C11_embedding/embedding_weightsBJinput_layer/sparse_input_layer/input_layer/C12_embedding/embedding_weightsBJinput_layer/sparse_input_layer/input_layer/C13_embedding/embedding_weightsBJinput_layer/sparse_input_layer/input_layer/C14_embedding/embedding_weightsBJinput_layer/sparse_input_layer/input_layer/C15_embedding/embedding_weightsBJinput_layer/sparse_input_layer/input_layer/C16_embedding/embedding_weightsBJinput_layer/sparse_input_layer/input_layer/C17_embedding/embedding_weightsBJinput_layer/sparse_input_layer/input_layer/C18_embedding/embedding_weightsBJinput_layer/sparse_input_layer/input_layer/C19_embedding/embedding_weightsBIinput_layer/sparse_input_layer/input_layer/C1_embedding/embedding_weightsBJinput_layer/sparse_input_layer/input_layer/C20_embedding/embedding_weightsBJinput_layer/sparse_input_layer/input_layer/C21_embedding/embedding_weightsBJinput_layer/sparse_input_layer/input_layer/C22_embedding/embedding_weightsBJinput_layer/sparse_input_layer/input_layer/C23_embedding/embedding_weightsBJinput_layer/sparse_input_layer/input_layer/C24_embedding/embedding_weightsBJinput_layer/sparse_input_layer/input_layer/C25_embedding/embedding_weightsBJinput_layer/sparse_input_layer/input_layer/C26_embedding/embedding_weightsBIinput_layer/sparse_input_layer/input_layer/C2_embedding/embedding_weightsBIinput_layer/sparse_input_layer/input_layer/C3_embedding/embedding_weightsBIinput_layer/sparse_input_layer/input_layer/C4_embedding/embedding_weightsBIinput_layer/sparse_input_layer/input_layer/C5_embedding/embedding_weightsBIinput_layer/sparse_input_layer/input_layer/C6_embedding/embedding_weightsBIinput_layer/sparse_input_layer/input_layer/C7_embedding/embedding_weightsBIinput_layer/sparse_input_layer/input_layer/C8_embedding/embedding_weightsBIinput_layer/sparse_input_layer/input_layer/C9_embedding/embedding_weightsBlogits/biasBlogits/kernelB<mlp_bot_layer/mlp_bot_hiddenlayer_0/batch_normalization/betaB=mlp_bot_layer/mlp_bot_hiddenlayer_0/batch_normalization/gammaBCmlp_bot_layer/mlp_bot_hiddenlayer_0/batch_normalization/moving_meanBGmlp_bot_layer/mlp_bot_hiddenlayer_0/batch_normalization/moving_varianceB(mlp_bot_layer/mlp_bot_hiddenlayer_0/biasB*mlp_bot_layer/mlp_bot_hiddenlayer_0/kernelB<mlp_bot_layer/mlp_bot_hiddenlayer_1/batch_normalization/betaB=mlp_bot_layer/mlp_bot_hiddenlayer_1/batch_normalization/gammaBCmlp_bot_layer/mlp_bot_hiddenlayer_1/batch_normalization/moving_meanBGmlp_bot_layer/mlp_bot_hiddenlayer_1/batch_normalization/moving_varianceB(mlp_bot_layer/mlp_bot_hiddenlayer_1/biasB*mlp_bot_layer/mlp_bot_hiddenlayer_1/kernelB<mlp_bot_layer/mlp_bot_hiddenlayer_2/batch_normalization/betaB=mlp_bot_layer/mlp_bot_hiddenlayer_2/batch_normalization/gammaBCmlp_bot_layer/mlp_bot_hiddenlayer_2/batch_normalization/moving_meanBGmlp_bot_layer/mlp_bot_hiddenlayer_2/batch_normalization/moving_varianceB(mlp_bot_layer/mlp_bot_hiddenlayer_2/biasB*mlp_bot_layer/mlp_bot_hiddenlayer_2/kernelB<mlp_bot_layer/mlp_bot_hiddenlayer_3/batch_normalization/betaB=mlp_bot_layer/mlp_bot_hiddenlayer_3/batch_normalization/gammaBCmlp_bot_layer/mlp_bot_hiddenlayer_3/batch_normalization/moving_meanBGmlp_bot_layer/mlp_bot_hiddenlayer_3/batch_normalization/moving_varianceB(mlp_bot_layer/mlp_bot_hiddenlayer_3/biasB*mlp_bot_layer/mlp_bot_hiddenlayer_3/kernelB(mlp_top_layer/mlp_top_hiddenlayer_0/biasB*mlp_top_layer/mlp_top_hiddenlayer_0/kernelB(mlp_top_layer/mlp_top_hiddenlayer_1/biasB*mlp_top_layer/mlp_top_hiddenlayer_1/kernel
∆
save/RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:9*
dtype0*г
valueўB÷9B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B	512 0,512B	512 0,512B B B	512 0,512B13 512 0,13:0,512B	256 0,256B	256 0,256B B B	256 0,256B512 256 0,512:0,256B64 0,64B64 0,64B B B64 0,64B256 64 0,256:0,64B16 0,16B16 0,16B B B16 0,16B64 16 0,64:0,16B	512 0,512B432 512 0,432:0,512B	256 0,256B432 256 0,432:0,256
К
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0* 
_output_shapesЈ
і::::::::::::::::::::::::::::::А:А:::А:	А:А:А:::А:
АА:@:@:::@:	А@::::::@:А:
∞А:А:
∞А*G
dtypes=
;29	
s
save/AssignAssignglobal_stepsave/RestoreV2*
_output_shapes
: *
_class
loc:@global_step*
T0	
ю
save/Assign_1AssignJinput_layer/sparse_input_layer/input_layer/C10_embedding/embedding_weightssave/RestoreV2:1*
_output_shapes
:	РN*]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C10_embedding/embedding_weights*
T0
ю
save/Assign_2AssignJinput_layer/sparse_input_layer/input_layer/C11_embedding/embedding_weightssave/RestoreV2:2*]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C11_embedding/embedding_weights*
_output_shapes
:	РN*
T0
ю
save/Assign_3AssignJinput_layer/sparse_input_layer/input_layer/C12_embedding/embedding_weightssave/RestoreV2:3*]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C12_embedding/embedding_weights*
_output_shapes
:	РN*
T0
ю
save/Assign_4AssignJinput_layer/sparse_input_layer/input_layer/C13_embedding/embedding_weightssave/RestoreV2:4*
T0*]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C13_embedding/embedding_weights*
_output_shapes
:	РN
ю
save/Assign_5AssignJinput_layer/sparse_input_layer/input_layer/C14_embedding/embedding_weightssave/RestoreV2:5*
T0*]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C14_embedding/embedding_weights*
_output_shapes
:	РN
ю
save/Assign_6AssignJinput_layer/sparse_input_layer/input_layer/C15_embedding/embedding_weightssave/RestoreV2:6*
_output_shapes
:	РN*]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C15_embedding/embedding_weights*
T0
ю
save/Assign_7AssignJinput_layer/sparse_input_layer/input_layer/C16_embedding/embedding_weightssave/RestoreV2:7*
_output_shapes
:	РN*]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C16_embedding/embedding_weights*
T0
ю
save/Assign_8AssignJinput_layer/sparse_input_layer/input_layer/C17_embedding/embedding_weightssave/RestoreV2:8*
_output_shapes
:	РN*
T0*]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C17_embedding/embedding_weights
ю
save/Assign_9AssignJinput_layer/sparse_input_layer/input_layer/C18_embedding/embedding_weightssave/RestoreV2:9*
_output_shapes
:	РN*]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C18_embedding/embedding_weights*
T0
А
save/Assign_10AssignJinput_layer/sparse_input_layer/input_layer/C19_embedding/embedding_weightssave/RestoreV2:10*
T0*
_output_shapes
:	РN*]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C19_embedding/embedding_weights
ю
save/Assign_11AssignIinput_layer/sparse_input_layer/input_layer/C1_embedding/embedding_weightssave/RestoreV2:11*
_output_shapes
:	РN*
T0*\
_classR
PNloc:@input_layer/sparse_input_layer/input_layer/C1_embedding/embedding_weights
А
save/Assign_12AssignJinput_layer/sparse_input_layer/input_layer/C20_embedding/embedding_weightssave/RestoreV2:12*
T0*
_output_shapes
:	РN*]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C20_embedding/embedding_weights
А
save/Assign_13AssignJinput_layer/sparse_input_layer/input_layer/C21_embedding/embedding_weightssave/RestoreV2:13*
T0*]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C21_embedding/embedding_weights*
_output_shapes
:	РN
А
save/Assign_14AssignJinput_layer/sparse_input_layer/input_layer/C22_embedding/embedding_weightssave/RestoreV2:14*]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C22_embedding/embedding_weights*
_output_shapes
:	РN*
T0
А
save/Assign_15AssignJinput_layer/sparse_input_layer/input_layer/C23_embedding/embedding_weightssave/RestoreV2:15*
_output_shapes
:	РN*]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C23_embedding/embedding_weights*
T0
А
save/Assign_16AssignJinput_layer/sparse_input_layer/input_layer/C24_embedding/embedding_weightssave/RestoreV2:16*]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C24_embedding/embedding_weights*
_output_shapes
:	РN*
T0
А
save/Assign_17AssignJinput_layer/sparse_input_layer/input_layer/C25_embedding/embedding_weightssave/RestoreV2:17*
_output_shapes
:	РN*
T0*]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C25_embedding/embedding_weights
А
save/Assign_18AssignJinput_layer/sparse_input_layer/input_layer/C26_embedding/embedding_weightssave/RestoreV2:18*]
_classS
QOloc:@input_layer/sparse_input_layer/input_layer/C26_embedding/embedding_weights*
T0*
_output_shapes
:	РN
ю
save/Assign_19AssignIinput_layer/sparse_input_layer/input_layer/C2_embedding/embedding_weightssave/RestoreV2:19*
T0*
_output_shapes
:	РN*\
_classR
PNloc:@input_layer/sparse_input_layer/input_layer/C2_embedding/embedding_weights
ю
save/Assign_20AssignIinput_layer/sparse_input_layer/input_layer/C3_embedding/embedding_weightssave/RestoreV2:20*
_output_shapes
:	РN*\
_classR
PNloc:@input_layer/sparse_input_layer/input_layer/C3_embedding/embedding_weights*
T0
ю
save/Assign_21AssignIinput_layer/sparse_input_layer/input_layer/C4_embedding/embedding_weightssave/RestoreV2:21*
_output_shapes
:	РN*\
_classR
PNloc:@input_layer/sparse_input_layer/input_layer/C4_embedding/embedding_weights*
T0
ю
save/Assign_22AssignIinput_layer/sparse_input_layer/input_layer/C5_embedding/embedding_weightssave/RestoreV2:22*\
_classR
PNloc:@input_layer/sparse_input_layer/input_layer/C5_embedding/embedding_weights*
_output_shapes
:	РN*
T0
ю
save/Assign_23AssignIinput_layer/sparse_input_layer/input_layer/C6_embedding/embedding_weightssave/RestoreV2:23*
_output_shapes
:	РN*
T0*\
_classR
PNloc:@input_layer/sparse_input_layer/input_layer/C6_embedding/embedding_weights
ю
save/Assign_24AssignIinput_layer/sparse_input_layer/input_layer/C7_embedding/embedding_weightssave/RestoreV2:24*
_output_shapes
:	РN*\
_classR
PNloc:@input_layer/sparse_input_layer/input_layer/C7_embedding/embedding_weights*
T0
ю
save/Assign_25AssignIinput_layer/sparse_input_layer/input_layer/C8_embedding/embedding_weightssave/RestoreV2:25*\
_classR
PNloc:@input_layer/sparse_input_layer/input_layer/C8_embedding/embedding_weights*
T0*
_output_shapes
:	РN
ю
save/Assign_26AssignIinput_layer/sparse_input_layer/input_layer/C9_embedding/embedding_weightssave/RestoreV2:26*
T0*
_output_shapes
:	РN*\
_classR
PNloc:@input_layer/sparse_input_layer/input_layer/C9_embedding/embedding_weights
}
save/Assign_27Assignlogits/biassave/RestoreV2:27*
_class
loc:@logits/bias*
T0*
_output_shapes
:
Ж
save/Assign_28Assignlogits/kernelsave/RestoreV2:28* 
_class
loc:@logits/kernel*
_output_shapes
:	А*
T0
о
save/Assign_29AssignCmlp_bot_layer/mlp_bot_hiddenlayer_0/batch_normalization/beta/part_0save/RestoreV2:29*V
_classL
JHloc:@mlp_bot_layer/mlp_bot_hiddenlayer_0/batch_normalization/beta/part_0*
T0*
_output_shapes	
:А
р
save/Assign_30AssignDmlp_bot_layer/mlp_bot_hiddenlayer_0/batch_normalization/gamma/part_0save/RestoreV2:30*W
_classM
KIloc:@mlp_bot_layer/mlp_bot_hiddenlayer_0/batch_normalization/gamma/part_0*
_output_shapes	
:А*
T0
о
save/Assign_31AssignCmlp_bot_layer/mlp_bot_hiddenlayer_0/batch_normalization/moving_meansave/RestoreV2:31*
T0*V
_classL
JHloc:@mlp_bot_layer/mlp_bot_hiddenlayer_0/batch_normalization/moving_mean*
_output_shapes	
:А
ц
save/Assign_32AssignGmlp_bot_layer/mlp_bot_hiddenlayer_0/batch_normalization/moving_variancesave/RestoreV2:32*
_output_shapes	
:А*
T0*Z
_classP
NLloc:@mlp_bot_layer/mlp_bot_hiddenlayer_0/batch_normalization/moving_variance
∆
save/Assign_33Assign/mlp_bot_layer/mlp_bot_hiddenlayer_0/bias/part_0save/RestoreV2:33*
_output_shapes	
:А*
T0*B
_class8
64loc:@mlp_bot_layer/mlp_bot_hiddenlayer_0/bias/part_0
ќ
save/Assign_34Assign1mlp_bot_layer/mlp_bot_hiddenlayer_0/kernel/part_0save/RestoreV2:34*
_output_shapes
:	А*
T0*D
_class:
86loc:@mlp_bot_layer/mlp_bot_hiddenlayer_0/kernel/part_0
о
save/Assign_35AssignCmlp_bot_layer/mlp_bot_hiddenlayer_1/batch_normalization/beta/part_0save/RestoreV2:35*V
_classL
JHloc:@mlp_bot_layer/mlp_bot_hiddenlayer_1/batch_normalization/beta/part_0*
_output_shapes	
:А*
T0
р
save/Assign_36AssignDmlp_bot_layer/mlp_bot_hiddenlayer_1/batch_normalization/gamma/part_0save/RestoreV2:36*
_output_shapes	
:А*W
_classM
KIloc:@mlp_bot_layer/mlp_bot_hiddenlayer_1/batch_normalization/gamma/part_0*
T0
о
save/Assign_37AssignCmlp_bot_layer/mlp_bot_hiddenlayer_1/batch_normalization/moving_meansave/RestoreV2:37*
T0*
_output_shapes	
:А*V
_classL
JHloc:@mlp_bot_layer/mlp_bot_hiddenlayer_1/batch_normalization/moving_mean
ц
save/Assign_38AssignGmlp_bot_layer/mlp_bot_hiddenlayer_1/batch_normalization/moving_variancesave/RestoreV2:38*
T0*Z
_classP
NLloc:@mlp_bot_layer/mlp_bot_hiddenlayer_1/batch_normalization/moving_variance*
_output_shapes	
:А
∆
save/Assign_39Assign/mlp_bot_layer/mlp_bot_hiddenlayer_1/bias/part_0save/RestoreV2:39*B
_class8
64loc:@mlp_bot_layer/mlp_bot_hiddenlayer_1/bias/part_0*
_output_shapes	
:А*
T0
ѕ
save/Assign_40Assign1mlp_bot_layer/mlp_bot_hiddenlayer_1/kernel/part_0save/RestoreV2:40*D
_class:
86loc:@mlp_bot_layer/mlp_bot_hiddenlayer_1/kernel/part_0* 
_output_shapes
:
АА*
T0
н
save/Assign_41AssignCmlp_bot_layer/mlp_bot_hiddenlayer_2/batch_normalization/beta/part_0save/RestoreV2:41*
_output_shapes
:@*
T0*V
_classL
JHloc:@mlp_bot_layer/mlp_bot_hiddenlayer_2/batch_normalization/beta/part_0
п
save/Assign_42AssignDmlp_bot_layer/mlp_bot_hiddenlayer_2/batch_normalization/gamma/part_0save/RestoreV2:42*W
_classM
KIloc:@mlp_bot_layer/mlp_bot_hiddenlayer_2/batch_normalization/gamma/part_0*
T0*
_output_shapes
:@
н
save/Assign_43AssignCmlp_bot_layer/mlp_bot_hiddenlayer_2/batch_normalization/moving_meansave/RestoreV2:43*
T0*V
_classL
JHloc:@mlp_bot_layer/mlp_bot_hiddenlayer_2/batch_normalization/moving_mean*
_output_shapes
:@
х
save/Assign_44AssignGmlp_bot_layer/mlp_bot_hiddenlayer_2/batch_normalization/moving_variancesave/RestoreV2:44*
_output_shapes
:@*Z
_classP
NLloc:@mlp_bot_layer/mlp_bot_hiddenlayer_2/batch_normalization/moving_variance*
T0
≈
save/Assign_45Assign/mlp_bot_layer/mlp_bot_hiddenlayer_2/bias/part_0save/RestoreV2:45*
_output_shapes
:@*
T0*B
_class8
64loc:@mlp_bot_layer/mlp_bot_hiddenlayer_2/bias/part_0
ќ
save/Assign_46Assign1mlp_bot_layer/mlp_bot_hiddenlayer_2/kernel/part_0save/RestoreV2:46*
T0*
_output_shapes
:	А@*D
_class:
86loc:@mlp_bot_layer/mlp_bot_hiddenlayer_2/kernel/part_0
н
save/Assign_47AssignCmlp_bot_layer/mlp_bot_hiddenlayer_3/batch_normalization/beta/part_0save/RestoreV2:47*V
_classL
JHloc:@mlp_bot_layer/mlp_bot_hiddenlayer_3/batch_normalization/beta/part_0*
_output_shapes
:*
T0
п
save/Assign_48AssignDmlp_bot_layer/mlp_bot_hiddenlayer_3/batch_normalization/gamma/part_0save/RestoreV2:48*W
_classM
KIloc:@mlp_bot_layer/mlp_bot_hiddenlayer_3/batch_normalization/gamma/part_0*
_output_shapes
:*
T0
н
save/Assign_49AssignCmlp_bot_layer/mlp_bot_hiddenlayer_3/batch_normalization/moving_meansave/RestoreV2:49*V
_classL
JHloc:@mlp_bot_layer/mlp_bot_hiddenlayer_3/batch_normalization/moving_mean*
_output_shapes
:*
T0
х
save/Assign_50AssignGmlp_bot_layer/mlp_bot_hiddenlayer_3/batch_normalization/moving_variancesave/RestoreV2:50*Z
_classP
NLloc:@mlp_bot_layer/mlp_bot_hiddenlayer_3/batch_normalization/moving_variance*
_output_shapes
:*
T0
≈
save/Assign_51Assign/mlp_bot_layer/mlp_bot_hiddenlayer_3/bias/part_0save/RestoreV2:51*B
_class8
64loc:@mlp_bot_layer/mlp_bot_hiddenlayer_3/bias/part_0*
T0*
_output_shapes
:
Ќ
save/Assign_52Assign1mlp_bot_layer/mlp_bot_hiddenlayer_3/kernel/part_0save/RestoreV2:52*D
_class:
86loc:@mlp_bot_layer/mlp_bot_hiddenlayer_3/kernel/part_0*
_output_shapes

:@*
T0
∆
save/Assign_53Assign/mlp_top_layer/mlp_top_hiddenlayer_0/bias/part_0save/RestoreV2:53*
T0*B
_class8
64loc:@mlp_top_layer/mlp_top_hiddenlayer_0/bias/part_0*
_output_shapes	
:А
ѕ
save/Assign_54Assign1mlp_top_layer/mlp_top_hiddenlayer_0/kernel/part_0save/RestoreV2:54* 
_output_shapes
:
∞А*
T0*D
_class:
86loc:@mlp_top_layer/mlp_top_hiddenlayer_0/kernel/part_0
∆
save/Assign_55Assign/mlp_top_layer/mlp_top_hiddenlayer_1/bias/part_0save/RestoreV2:55*B
_class8
64loc:@mlp_top_layer/mlp_top_hiddenlayer_1/bias/part_0*
T0*
_output_shapes	
:А
ѕ
save/Assign_56Assign1mlp_top_layer/mlp_top_hiddenlayer_1/kernel/part_0save/RestoreV2:56* 
_output_shapes
:
∞А*D
_class:
86loc:@mlp_top_layer/mlp_top_hiddenlayer_1/kernel/part_0*
T0
„
save/restore_shardNoOp^save/Assign^save/Assign_1^save/Assign_10^save/Assign_11^save/Assign_12^save/Assign_13^save/Assign_14^save/Assign_15^save/Assign_16^save/Assign_17^save/Assign_18^save/Assign_19^save/Assign_2^save/Assign_20^save/Assign_21^save/Assign_22^save/Assign_23^save/Assign_24^save/Assign_25^save/Assign_26^save/Assign_27^save/Assign_28^save/Assign_29^save/Assign_3^save/Assign_30^save/Assign_31^save/Assign_32^save/Assign_33^save/Assign_34^save/Assign_35^save/Assign_36^save/Assign_37^save/Assign_38^save/Assign_39^save/Assign_4^save/Assign_40^save/Assign_41^save/Assign_42^save/Assign_43^save/Assign_44^save/Assign_45^save/Assign_46^save/Assign_47^save/Assign_48^save/Assign_49^save/Assign_5^save/Assign_50^save/Assign_51^save/Assign_52^save/Assign_53^save/Assign_54^save/Assign_55^save/Assign_56^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9
-
save/restore_allNoOp^save/restore_shard"Ж<
save/Const:0save/Identity:0save/restore_all (5 @F8"мУ
	variablesЁУўУ
Z
global_step:0global_step/Assignglobal_step/read:02global_step/Initializer/zeros:0H
б
Linput_layer/sparse_input_layer/input_layer/C10_embedding/embedding_weights:0Qinput_layer/sparse_input_layer/input_layer/C10_embedding/embedding_weights/AssignQinput_layer/sparse_input_layer/input_layer/C10_embedding/embedding_weights/read:02iinput_layer/sparse_input_layer/input_layer/C10_embedding/embedding_weights/Initializer/truncated_normal:08
б
Linput_layer/sparse_input_layer/input_layer/C11_embedding/embedding_weights:0Qinput_layer/sparse_input_layer/input_layer/C11_embedding/embedding_weights/AssignQinput_layer/sparse_input_layer/input_layer/C11_embedding/embedding_weights/read:02iinput_layer/sparse_input_layer/input_layer/C11_embedding/embedding_weights/Initializer/truncated_normal:08
б
Linput_layer/sparse_input_layer/input_layer/C12_embedding/embedding_weights:0Qinput_layer/sparse_input_layer/input_layer/C12_embedding/embedding_weights/AssignQinput_layer/sparse_input_layer/input_layer/C12_embedding/embedding_weights/read:02iinput_layer/sparse_input_layer/input_layer/C12_embedding/embedding_weights/Initializer/truncated_normal:08
б
Linput_layer/sparse_input_layer/input_layer/C13_embedding/embedding_weights:0Qinput_layer/sparse_input_layer/input_layer/C13_embedding/embedding_weights/AssignQinput_layer/sparse_input_layer/input_layer/C13_embedding/embedding_weights/read:02iinput_layer/sparse_input_layer/input_layer/C13_embedding/embedding_weights/Initializer/truncated_normal:08
б
Linput_layer/sparse_input_layer/input_layer/C14_embedding/embedding_weights:0Qinput_layer/sparse_input_layer/input_layer/C14_embedding/embedding_weights/AssignQinput_layer/sparse_input_layer/input_layer/C14_embedding/embedding_weights/read:02iinput_layer/sparse_input_layer/input_layer/C14_embedding/embedding_weights/Initializer/truncated_normal:08
б
Linput_layer/sparse_input_layer/input_layer/C15_embedding/embedding_weights:0Qinput_layer/sparse_input_layer/input_layer/C15_embedding/embedding_weights/AssignQinput_layer/sparse_input_layer/input_layer/C15_embedding/embedding_weights/read:02iinput_layer/sparse_input_layer/input_layer/C15_embedding/embedding_weights/Initializer/truncated_normal:08
б
Linput_layer/sparse_input_layer/input_layer/C16_embedding/embedding_weights:0Qinput_layer/sparse_input_layer/input_layer/C16_embedding/embedding_weights/AssignQinput_layer/sparse_input_layer/input_layer/C16_embedding/embedding_weights/read:02iinput_layer/sparse_input_layer/input_layer/C16_embedding/embedding_weights/Initializer/truncated_normal:08
б
Linput_layer/sparse_input_layer/input_layer/C17_embedding/embedding_weights:0Qinput_layer/sparse_input_layer/input_layer/C17_embedding/embedding_weights/AssignQinput_layer/sparse_input_layer/input_layer/C17_embedding/embedding_weights/read:02iinput_layer/sparse_input_layer/input_layer/C17_embedding/embedding_weights/Initializer/truncated_normal:08
б
Linput_layer/sparse_input_layer/input_layer/C18_embedding/embedding_weights:0Qinput_layer/sparse_input_layer/input_layer/C18_embedding/embedding_weights/AssignQinput_layer/sparse_input_layer/input_layer/C18_embedding/embedding_weights/read:02iinput_layer/sparse_input_layer/input_layer/C18_embedding/embedding_weights/Initializer/truncated_normal:08
б
Linput_layer/sparse_input_layer/input_layer/C19_embedding/embedding_weights:0Qinput_layer/sparse_input_layer/input_layer/C19_embedding/embedding_weights/AssignQinput_layer/sparse_input_layer/input_layer/C19_embedding/embedding_weights/read:02iinput_layer/sparse_input_layer/input_layer/C19_embedding/embedding_weights/Initializer/truncated_normal:08
Ё
Kinput_layer/sparse_input_layer/input_layer/C1_embedding/embedding_weights:0Pinput_layer/sparse_input_layer/input_layer/C1_embedding/embedding_weights/AssignPinput_layer/sparse_input_layer/input_layer/C1_embedding/embedding_weights/read:02hinput_layer/sparse_input_layer/input_layer/C1_embedding/embedding_weights/Initializer/truncated_normal:08
б
Linput_layer/sparse_input_layer/input_layer/C20_embedding/embedding_weights:0Qinput_layer/sparse_input_layer/input_layer/C20_embedding/embedding_weights/AssignQinput_layer/sparse_input_layer/input_layer/C20_embedding/embedding_weights/read:02iinput_layer/sparse_input_layer/input_layer/C20_embedding/embedding_weights/Initializer/truncated_normal:08
б
Linput_layer/sparse_input_layer/input_layer/C21_embedding/embedding_weights:0Qinput_layer/sparse_input_layer/input_layer/C21_embedding/embedding_weights/AssignQinput_layer/sparse_input_layer/input_layer/C21_embedding/embedding_weights/read:02iinput_layer/sparse_input_layer/input_layer/C21_embedding/embedding_weights/Initializer/truncated_normal:08
б
Linput_layer/sparse_input_layer/input_layer/C22_embedding/embedding_weights:0Qinput_layer/sparse_input_layer/input_layer/C22_embedding/embedding_weights/AssignQinput_layer/sparse_input_layer/input_layer/C22_embedding/embedding_weights/read:02iinput_layer/sparse_input_layer/input_layer/C22_embedding/embedding_weights/Initializer/truncated_normal:08
б
Linput_layer/sparse_input_layer/input_layer/C23_embedding/embedding_weights:0Qinput_layer/sparse_input_layer/input_layer/C23_embedding/embedding_weights/AssignQinput_layer/sparse_input_layer/input_layer/C23_embedding/embedding_weights/read:02iinput_layer/sparse_input_layer/input_layer/C23_embedding/embedding_weights/Initializer/truncated_normal:08
б
Linput_layer/sparse_input_layer/input_layer/C24_embedding/embedding_weights:0Qinput_layer/sparse_input_layer/input_layer/C24_embedding/embedding_weights/AssignQinput_layer/sparse_input_layer/input_layer/C24_embedding/embedding_weights/read:02iinput_layer/sparse_input_layer/input_layer/C24_embedding/embedding_weights/Initializer/truncated_normal:08
б
Linput_layer/sparse_input_layer/input_layer/C25_embedding/embedding_weights:0Qinput_layer/sparse_input_layer/input_layer/C25_embedding/embedding_weights/AssignQinput_layer/sparse_input_layer/input_layer/C25_embedding/embedding_weights/read:02iinput_layer/sparse_input_layer/input_layer/C25_embedding/embedding_weights/Initializer/truncated_normal:08
б
Linput_layer/sparse_input_layer/input_layer/C26_embedding/embedding_weights:0Qinput_layer/sparse_input_layer/input_layer/C26_embedding/embedding_weights/AssignQinput_layer/sparse_input_layer/input_layer/C26_embedding/embedding_weights/read:02iinput_layer/sparse_input_layer/input_layer/C26_embedding/embedding_weights/Initializer/truncated_normal:08
Ё
Kinput_layer/sparse_input_layer/input_layer/C2_embedding/embedding_weights:0Pinput_layer/sparse_input_layer/input_layer/C2_embedding/embedding_weights/AssignPinput_layer/sparse_input_layer/input_layer/C2_embedding/embedding_weights/read:02hinput_layer/sparse_input_layer/input_layer/C2_embedding/embedding_weights/Initializer/truncated_normal:08
Ё
Kinput_layer/sparse_input_layer/input_layer/C3_embedding/embedding_weights:0Pinput_layer/sparse_input_layer/input_layer/C3_embedding/embedding_weights/AssignPinput_layer/sparse_input_layer/input_layer/C3_embedding/embedding_weights/read:02hinput_layer/sparse_input_layer/input_layer/C3_embedding/embedding_weights/Initializer/truncated_normal:08
Ё
Kinput_layer/sparse_input_layer/input_layer/C4_embedding/embedding_weights:0Pinput_layer/sparse_input_layer/input_layer/C4_embedding/embedding_weights/AssignPinput_layer/sparse_input_layer/input_layer/C4_embedding/embedding_weights/read:02hinput_layer/sparse_input_layer/input_layer/C4_embedding/embedding_weights/Initializer/truncated_normal:08
Ё
Kinput_layer/sparse_input_layer/input_layer/C5_embedding/embedding_weights:0Pinput_layer/sparse_input_layer/input_layer/C5_embedding/embedding_weights/AssignPinput_layer/sparse_input_layer/input_layer/C5_embedding/embedding_weights/read:02hinput_layer/sparse_input_layer/input_layer/C5_embedding/embedding_weights/Initializer/truncated_normal:08
Ё
Kinput_layer/sparse_input_layer/input_layer/C6_embedding/embedding_weights:0Pinput_layer/sparse_input_layer/input_layer/C6_embedding/embedding_weights/AssignPinput_layer/sparse_input_layer/input_layer/C6_embedding/embedding_weights/read:02hinput_layer/sparse_input_layer/input_layer/C6_embedding/embedding_weights/Initializer/truncated_normal:08
Ё
Kinput_layer/sparse_input_layer/input_layer/C7_embedding/embedding_weights:0Pinput_layer/sparse_input_layer/input_layer/C7_embedding/embedding_weights/AssignPinput_layer/sparse_input_layer/input_layer/C7_embedding/embedding_weights/read:02hinput_layer/sparse_input_layer/input_layer/C7_embedding/embedding_weights/Initializer/truncated_normal:08
Ё
Kinput_layer/sparse_input_layer/input_layer/C8_embedding/embedding_weights:0Pinput_layer/sparse_input_layer/input_layer/C8_embedding/embedding_weights/AssignPinput_layer/sparse_input_layer/input_layer/C8_embedding/embedding_weights/read:02hinput_layer/sparse_input_layer/input_layer/C8_embedding/embedding_weights/Initializer/truncated_normal:08
Ё
Kinput_layer/sparse_input_layer/input_layer/C9_embedding/embedding_weights:0Pinput_layer/sparse_input_layer/input_layer/C9_embedding/embedding_weights/AssignPinput_layer/sparse_input_layer/input_layer/C9_embedding/embedding_weights/read:02hinput_layer/sparse_input_layer/input_layer/C9_embedding/embedding_weights/Initializer/truncated_normal:08
Ј
3mlp_bot_layer/mlp_bot_hiddenlayer_0/kernel/part_0:08mlp_bot_layer/mlp_bot_hiddenlayer_0/kernel/part_0/Assign8mlp_bot_layer/mlp_bot_hiddenlayer_0/kernel/part_0/read:0":
*mlp_bot_layer/mlp_bot_hiddenlayer_0/kernelА  "А2Nmlp_bot_layer/mlp_bot_hiddenlayer_0/kernel/part_0/Initializer/random_uniform:08
°
1mlp_bot_layer/mlp_bot_hiddenlayer_0/bias/part_0:06mlp_bot_layer/mlp_bot_hiddenlayer_0/bias/part_0/Assign6mlp_bot_layer/mlp_bot_hiddenlayer_0/bias/part_0/read:0"5
(mlp_bot_layer/mlp_bot_hiddenlayer_0/biasА "А2Cmlp_bot_layer/mlp_bot_hiddenlayer_0/bias/part_0/Initializer/zeros:08
Й
Fmlp_bot_layer/mlp_bot_hiddenlayer_0/batch_normalization/gamma/part_0:0Kmlp_bot_layer/mlp_bot_hiddenlayer_0/batch_normalization/gamma/part_0/AssignKmlp_bot_layer/mlp_bot_hiddenlayer_0/batch_normalization/gamma/part_0/read:0"J
=mlp_bot_layer/mlp_bot_hiddenlayer_0/batch_normalization/gammaА "А2Wmlp_bot_layer/mlp_bot_hiddenlayer_0/batch_normalization/gamma/part_0/Initializer/ones:08
Е
Emlp_bot_layer/mlp_bot_hiddenlayer_0/batch_normalization/beta/part_0:0Jmlp_bot_layer/mlp_bot_hiddenlayer_0/batch_normalization/beta/part_0/AssignJmlp_bot_layer/mlp_bot_hiddenlayer_0/batch_normalization/beta/part_0/read:0"I
<mlp_bot_layer/mlp_bot_hiddenlayer_0/batch_normalization/betaА "А2Wmlp_bot_layer/mlp_bot_hiddenlayer_0/batch_normalization/beta/part_0/Initializer/zeros:08
Љ
Emlp_bot_layer/mlp_bot_hiddenlayer_0/batch_normalization/moving_mean:0Jmlp_bot_layer/mlp_bot_hiddenlayer_0/batch_normalization/moving_mean/AssignJmlp_bot_layer/mlp_bot_hiddenlayer_0/batch_normalization/moving_mean/read:02Wmlp_bot_layer/mlp_bot_hiddenlayer_0/batch_normalization/moving_mean/Initializer/zeros:0@H
Ћ
Imlp_bot_layer/mlp_bot_hiddenlayer_0/batch_normalization/moving_variance:0Nmlp_bot_layer/mlp_bot_hiddenlayer_0/batch_normalization/moving_variance/AssignNmlp_bot_layer/mlp_bot_hiddenlayer_0/batch_normalization/moving_variance/read:02Zmlp_bot_layer/mlp_bot_hiddenlayer_0/batch_normalization/moving_variance/Initializer/ones:0@H
є
3mlp_bot_layer/mlp_bot_hiddenlayer_1/kernel/part_0:08mlp_bot_layer/mlp_bot_hiddenlayer_1/kernel/part_0/Assign8mlp_bot_layer/mlp_bot_hiddenlayer_1/kernel/part_0/read:0"<
*mlp_bot_layer/mlp_bot_hiddenlayer_1/kernelАА  "АА2Nmlp_bot_layer/mlp_bot_hiddenlayer_1/kernel/part_0/Initializer/random_uniform:08
°
1mlp_bot_layer/mlp_bot_hiddenlayer_1/bias/part_0:06mlp_bot_layer/mlp_bot_hiddenlayer_1/bias/part_0/Assign6mlp_bot_layer/mlp_bot_hiddenlayer_1/bias/part_0/read:0"5
(mlp_bot_layer/mlp_bot_hiddenlayer_1/biasА "А2Cmlp_bot_layer/mlp_bot_hiddenlayer_1/bias/part_0/Initializer/zeros:08
Й
Fmlp_bot_layer/mlp_bot_hiddenlayer_1/batch_normalization/gamma/part_0:0Kmlp_bot_layer/mlp_bot_hiddenlayer_1/batch_normalization/gamma/part_0/AssignKmlp_bot_layer/mlp_bot_hiddenlayer_1/batch_normalization/gamma/part_0/read:0"J
=mlp_bot_layer/mlp_bot_hiddenlayer_1/batch_normalization/gammaА "А2Wmlp_bot_layer/mlp_bot_hiddenlayer_1/batch_normalization/gamma/part_0/Initializer/ones:08
Е
Emlp_bot_layer/mlp_bot_hiddenlayer_1/batch_normalization/beta/part_0:0Jmlp_bot_layer/mlp_bot_hiddenlayer_1/batch_normalization/beta/part_0/AssignJmlp_bot_layer/mlp_bot_hiddenlayer_1/batch_normalization/beta/part_0/read:0"I
<mlp_bot_layer/mlp_bot_hiddenlayer_1/batch_normalization/betaА "А2Wmlp_bot_layer/mlp_bot_hiddenlayer_1/batch_normalization/beta/part_0/Initializer/zeros:08
Љ
Emlp_bot_layer/mlp_bot_hiddenlayer_1/batch_normalization/moving_mean:0Jmlp_bot_layer/mlp_bot_hiddenlayer_1/batch_normalization/moving_mean/AssignJmlp_bot_layer/mlp_bot_hiddenlayer_1/batch_normalization/moving_mean/read:02Wmlp_bot_layer/mlp_bot_hiddenlayer_1/batch_normalization/moving_mean/Initializer/zeros:0@H
Ћ
Imlp_bot_layer/mlp_bot_hiddenlayer_1/batch_normalization/moving_variance:0Nmlp_bot_layer/mlp_bot_hiddenlayer_1/batch_normalization/moving_variance/AssignNmlp_bot_layer/mlp_bot_hiddenlayer_1/batch_normalization/moving_variance/read:02Zmlp_bot_layer/mlp_bot_hiddenlayer_1/batch_normalization/moving_variance/Initializer/ones:0@H
Ј
3mlp_bot_layer/mlp_bot_hiddenlayer_2/kernel/part_0:08mlp_bot_layer/mlp_bot_hiddenlayer_2/kernel/part_0/Assign8mlp_bot_layer/mlp_bot_hiddenlayer_2/kernel/part_0/read:0":
*mlp_bot_layer/mlp_bot_hiddenlayer_2/kernelА@  "А@2Nmlp_bot_layer/mlp_bot_hiddenlayer_2/kernel/part_0/Initializer/random_uniform:08
Я
1mlp_bot_layer/mlp_bot_hiddenlayer_2/bias/part_0:06mlp_bot_layer/mlp_bot_hiddenlayer_2/bias/part_0/Assign6mlp_bot_layer/mlp_bot_hiddenlayer_2/bias/part_0/read:0"3
(mlp_bot_layer/mlp_bot_hiddenlayer_2/bias@ "@2Cmlp_bot_layer/mlp_bot_hiddenlayer_2/bias/part_0/Initializer/zeros:08
З
Fmlp_bot_layer/mlp_bot_hiddenlayer_2/batch_normalization/gamma/part_0:0Kmlp_bot_layer/mlp_bot_hiddenlayer_2/batch_normalization/gamma/part_0/AssignKmlp_bot_layer/mlp_bot_hiddenlayer_2/batch_normalization/gamma/part_0/read:0"H
=mlp_bot_layer/mlp_bot_hiddenlayer_2/batch_normalization/gamma@ "@2Wmlp_bot_layer/mlp_bot_hiddenlayer_2/batch_normalization/gamma/part_0/Initializer/ones:08
Г
Emlp_bot_layer/mlp_bot_hiddenlayer_2/batch_normalization/beta/part_0:0Jmlp_bot_layer/mlp_bot_hiddenlayer_2/batch_normalization/beta/part_0/AssignJmlp_bot_layer/mlp_bot_hiddenlayer_2/batch_normalization/beta/part_0/read:0"G
<mlp_bot_layer/mlp_bot_hiddenlayer_2/batch_normalization/beta@ "@2Wmlp_bot_layer/mlp_bot_hiddenlayer_2/batch_normalization/beta/part_0/Initializer/zeros:08
Љ
Emlp_bot_layer/mlp_bot_hiddenlayer_2/batch_normalization/moving_mean:0Jmlp_bot_layer/mlp_bot_hiddenlayer_2/batch_normalization/moving_mean/AssignJmlp_bot_layer/mlp_bot_hiddenlayer_2/batch_normalization/moving_mean/read:02Wmlp_bot_layer/mlp_bot_hiddenlayer_2/batch_normalization/moving_mean/Initializer/zeros:0@H
Ћ
Imlp_bot_layer/mlp_bot_hiddenlayer_2/batch_normalization/moving_variance:0Nmlp_bot_layer/mlp_bot_hiddenlayer_2/batch_normalization/moving_variance/AssignNmlp_bot_layer/mlp_bot_hiddenlayer_2/batch_normalization/moving_variance/read:02Zmlp_bot_layer/mlp_bot_hiddenlayer_2/batch_normalization/moving_variance/Initializer/ones:0@H
µ
3mlp_bot_layer/mlp_bot_hiddenlayer_3/kernel/part_0:08mlp_bot_layer/mlp_bot_hiddenlayer_3/kernel/part_0/Assign8mlp_bot_layer/mlp_bot_hiddenlayer_3/kernel/part_0/read:0"8
*mlp_bot_layer/mlp_bot_hiddenlayer_3/kernel@  "@2Nmlp_bot_layer/mlp_bot_hiddenlayer_3/kernel/part_0/Initializer/random_uniform:08
Я
1mlp_bot_layer/mlp_bot_hiddenlayer_3/bias/part_0:06mlp_bot_layer/mlp_bot_hiddenlayer_3/bias/part_0/Assign6mlp_bot_layer/mlp_bot_hiddenlayer_3/bias/part_0/read:0"3
(mlp_bot_layer/mlp_bot_hiddenlayer_3/bias "2Cmlp_bot_layer/mlp_bot_hiddenlayer_3/bias/part_0/Initializer/zeros:08
З
Fmlp_bot_layer/mlp_bot_hiddenlayer_3/batch_normalization/gamma/part_0:0Kmlp_bot_layer/mlp_bot_hiddenlayer_3/batch_normalization/gamma/part_0/AssignKmlp_bot_layer/mlp_bot_hiddenlayer_3/batch_normalization/gamma/part_0/read:0"H
=mlp_bot_layer/mlp_bot_hiddenlayer_3/batch_normalization/gamma "2Wmlp_bot_layer/mlp_bot_hiddenlayer_3/batch_normalization/gamma/part_0/Initializer/ones:08
Г
Emlp_bot_layer/mlp_bot_hiddenlayer_3/batch_normalization/beta/part_0:0Jmlp_bot_layer/mlp_bot_hiddenlayer_3/batch_normalization/beta/part_0/AssignJmlp_bot_layer/mlp_bot_hiddenlayer_3/batch_normalization/beta/part_0/read:0"G
<mlp_bot_layer/mlp_bot_hiddenlayer_3/batch_normalization/beta "2Wmlp_bot_layer/mlp_bot_hiddenlayer_3/batch_normalization/beta/part_0/Initializer/zeros:08
Љ
Emlp_bot_layer/mlp_bot_hiddenlayer_3/batch_normalization/moving_mean:0Jmlp_bot_layer/mlp_bot_hiddenlayer_3/batch_normalization/moving_mean/AssignJmlp_bot_layer/mlp_bot_hiddenlayer_3/batch_normalization/moving_mean/read:02Wmlp_bot_layer/mlp_bot_hiddenlayer_3/batch_normalization/moving_mean/Initializer/zeros:0@H
Ћ
Imlp_bot_layer/mlp_bot_hiddenlayer_3/batch_normalization/moving_variance:0Nmlp_bot_layer/mlp_bot_hiddenlayer_3/batch_normalization/moving_variance/AssignNmlp_bot_layer/mlp_bot_hiddenlayer_3/batch_normalization/moving_variance/read:02Zmlp_bot_layer/mlp_bot_hiddenlayer_3/batch_normalization/moving_variance/Initializer/ones:0@H
є
3mlp_top_layer/mlp_top_hiddenlayer_0/kernel/part_0:08mlp_top_layer/mlp_top_hiddenlayer_0/kernel/part_0/Assign8mlp_top_layer/mlp_top_hiddenlayer_0/kernel/part_0/read:0"<
*mlp_top_layer/mlp_top_hiddenlayer_0/kernel∞А  "∞А2Nmlp_top_layer/mlp_top_hiddenlayer_0/kernel/part_0/Initializer/random_uniform:08
°
1mlp_top_layer/mlp_top_hiddenlayer_0/bias/part_0:06mlp_top_layer/mlp_top_hiddenlayer_0/bias/part_0/Assign6mlp_top_layer/mlp_top_hiddenlayer_0/bias/part_0/read:0"5
(mlp_top_layer/mlp_top_hiddenlayer_0/biasА "А2Cmlp_top_layer/mlp_top_hiddenlayer_0/bias/part_0/Initializer/zeros:08
є
3mlp_top_layer/mlp_top_hiddenlayer_1/kernel/part_0:08mlp_top_layer/mlp_top_hiddenlayer_1/kernel/part_0/Assign8mlp_top_layer/mlp_top_hiddenlayer_1/kernel/part_0/read:0"<
*mlp_top_layer/mlp_top_hiddenlayer_1/kernel∞А  "∞А2Nmlp_top_layer/mlp_top_hiddenlayer_1/kernel/part_0/Initializer/random_uniform:08
°
1mlp_top_layer/mlp_top_hiddenlayer_1/bias/part_0:06mlp_top_layer/mlp_top_hiddenlayer_1/bias/part_0/Assign6mlp_top_layer/mlp_top_hiddenlayer_1/bias/part_0/read:0"5
(mlp_top_layer/mlp_top_hiddenlayer_1/biasА "А2Cmlp_top_layer/mlp_top_hiddenlayer_1/bias/part_0/Initializer/zeros:08
k
logits/kernel:0logits/kernel/Assignlogits/kernel/read:02*logits/kernel/Initializer/random_uniform:08
Z
logits/bias:0logits/bias/Assignlogits/bias/read:02logits/bias/Initializer/zeros:08"в

update_ops”
–
Gmlp_bot_layer/mlp_bot_hiddenlayer_0/batch_normalization/AssignMovingAvg
Imlp_bot_layer/mlp_bot_hiddenlayer_0/batch_normalization/AssignMovingAvg_1
Gmlp_bot_layer/mlp_bot_hiddenlayer_1/batch_normalization/AssignMovingAvg
Imlp_bot_layer/mlp_bot_hiddenlayer_1/batch_normalization/AssignMovingAvg_1
Gmlp_bot_layer/mlp_bot_hiddenlayer_2/batch_normalization/AssignMovingAvg
Imlp_bot_layer/mlp_bot_hiddenlayer_2/batch_normalization/AssignMovingAvg_1
Gmlp_bot_layer/mlp_bot_hiddenlayer_3/batch_normalization/AssignMovingAvg
Imlp_bot_layer/mlp_bot_hiddenlayer_3/batch_normalization/AssignMovingAvg_1"%
saved_model_main_op


group_deps"‘
	summaries∆
√
amlp_bot_layer/mlp_bot_hiddenlayer_0/mlp_bot_layer/mlp_bot_hiddenlayer_0/fraction_of_zero_values:0
Tmlp_bot_layer/mlp_bot_hiddenlayer_0/mlp_bot_layer/mlp_bot_hiddenlayer_0/activation:0
amlp_bot_layer/mlp_bot_hiddenlayer_1/mlp_bot_layer/mlp_bot_hiddenlayer_1/fraction_of_zero_values:0
Tmlp_bot_layer/mlp_bot_hiddenlayer_1/mlp_bot_layer/mlp_bot_hiddenlayer_1/activation:0
amlp_bot_layer/mlp_bot_hiddenlayer_2/mlp_bot_layer/mlp_bot_hiddenlayer_2/fraction_of_zero_values:0
Tmlp_bot_layer/mlp_bot_hiddenlayer_2/mlp_bot_layer/mlp_bot_hiddenlayer_2/activation:0
amlp_bot_layer/mlp_bot_hiddenlayer_3/mlp_bot_layer/mlp_bot_hiddenlayer_3/fraction_of_zero_values:0
Tmlp_bot_layer/mlp_bot_hiddenlayer_3/mlp_bot_layer/mlp_bot_hiddenlayer_3/activation:0
Kmlp_top_layer/mlp_top_layer/mlp_top_hiddenlayer_0/fraction_of_zero_values:0
>mlp_top_layer/mlp_top_layer/mlp_top_hiddenlayer_0/activation:0
Kmlp_top_layer/mlp_top_layer/mlp_top_hiddenlayer_1/fraction_of_zero_values:0
>mlp_top_layer/mlp_top_layer/mlp_top_hiddenlayer_1/activation:0
'logits/logits/fraction_of_zero_values:0
logits/logits/activation:0"m
global_step^\
Z
global_step:0global_step/Assignglobal_step/read:02global_step/Initializer/zeros:0H"ЫH
model_variablesЗHДH
б
Linput_layer/sparse_input_layer/input_layer/C10_embedding/embedding_weights:0Qinput_layer/sparse_input_layer/input_layer/C10_embedding/embedding_weights/AssignQinput_layer/sparse_input_layer/input_layer/C10_embedding/embedding_weights/read:02iinput_layer/sparse_input_layer/input_layer/C10_embedding/embedding_weights/Initializer/truncated_normal:08
б
Linput_layer/sparse_input_layer/input_layer/C11_embedding/embedding_weights:0Qinput_layer/sparse_input_layer/input_layer/C11_embedding/embedding_weights/AssignQinput_layer/sparse_input_layer/input_layer/C11_embedding/embedding_weights/read:02iinput_layer/sparse_input_layer/input_layer/C11_embedding/embedding_weights/Initializer/truncated_normal:08
б
Linput_layer/sparse_input_layer/input_layer/C12_embedding/embedding_weights:0Qinput_layer/sparse_input_layer/input_layer/C12_embedding/embedding_weights/AssignQinput_layer/sparse_input_layer/input_layer/C12_embedding/embedding_weights/read:02iinput_layer/sparse_input_layer/input_layer/C12_embedding/embedding_weights/Initializer/truncated_normal:08
б
Linput_layer/sparse_input_layer/input_layer/C13_embedding/embedding_weights:0Qinput_layer/sparse_input_layer/input_layer/C13_embedding/embedding_weights/AssignQinput_layer/sparse_input_layer/input_layer/C13_embedding/embedding_weights/read:02iinput_layer/sparse_input_layer/input_layer/C13_embedding/embedding_weights/Initializer/truncated_normal:08
б
Linput_layer/sparse_input_layer/input_layer/C14_embedding/embedding_weights:0Qinput_layer/sparse_input_layer/input_layer/C14_embedding/embedding_weights/AssignQinput_layer/sparse_input_layer/input_layer/C14_embedding/embedding_weights/read:02iinput_layer/sparse_input_layer/input_layer/C14_embedding/embedding_weights/Initializer/truncated_normal:08
б
Linput_layer/sparse_input_layer/input_layer/C15_embedding/embedding_weights:0Qinput_layer/sparse_input_layer/input_layer/C15_embedding/embedding_weights/AssignQinput_layer/sparse_input_layer/input_layer/C15_embedding/embedding_weights/read:02iinput_layer/sparse_input_layer/input_layer/C15_embedding/embedding_weights/Initializer/truncated_normal:08
б
Linput_layer/sparse_input_layer/input_layer/C16_embedding/embedding_weights:0Qinput_layer/sparse_input_layer/input_layer/C16_embedding/embedding_weights/AssignQinput_layer/sparse_input_layer/input_layer/C16_embedding/embedding_weights/read:02iinput_layer/sparse_input_layer/input_layer/C16_embedding/embedding_weights/Initializer/truncated_normal:08
б
Linput_layer/sparse_input_layer/input_layer/C17_embedding/embedding_weights:0Qinput_layer/sparse_input_layer/input_layer/C17_embedding/embedding_weights/AssignQinput_layer/sparse_input_layer/input_layer/C17_embedding/embedding_weights/read:02iinput_layer/sparse_input_layer/input_layer/C17_embedding/embedding_weights/Initializer/truncated_normal:08
б
Linput_layer/sparse_input_layer/input_layer/C18_embedding/embedding_weights:0Qinput_layer/sparse_input_layer/input_layer/C18_embedding/embedding_weights/AssignQinput_layer/sparse_input_layer/input_layer/C18_embedding/embedding_weights/read:02iinput_layer/sparse_input_layer/input_layer/C18_embedding/embedding_weights/Initializer/truncated_normal:08
б
Linput_layer/sparse_input_layer/input_layer/C19_embedding/embedding_weights:0Qinput_layer/sparse_input_layer/input_layer/C19_embedding/embedding_weights/AssignQinput_layer/sparse_input_layer/input_layer/C19_embedding/embedding_weights/read:02iinput_layer/sparse_input_layer/input_layer/C19_embedding/embedding_weights/Initializer/truncated_normal:08
Ё
Kinput_layer/sparse_input_layer/input_layer/C1_embedding/embedding_weights:0Pinput_layer/sparse_input_layer/input_layer/C1_embedding/embedding_weights/AssignPinput_layer/sparse_input_layer/input_layer/C1_embedding/embedding_weights/read:02hinput_layer/sparse_input_layer/input_layer/C1_embedding/embedding_weights/Initializer/truncated_normal:08
б
Linput_layer/sparse_input_layer/input_layer/C20_embedding/embedding_weights:0Qinput_layer/sparse_input_layer/input_layer/C20_embedding/embedding_weights/AssignQinput_layer/sparse_input_layer/input_layer/C20_embedding/embedding_weights/read:02iinput_layer/sparse_input_layer/input_layer/C20_embedding/embedding_weights/Initializer/truncated_normal:08
б
Linput_layer/sparse_input_layer/input_layer/C21_embedding/embedding_weights:0Qinput_layer/sparse_input_layer/input_layer/C21_embedding/embedding_weights/AssignQinput_layer/sparse_input_layer/input_layer/C21_embedding/embedding_weights/read:02iinput_layer/sparse_input_layer/input_layer/C21_embedding/embedding_weights/Initializer/truncated_normal:08
б
Linput_layer/sparse_input_layer/input_layer/C22_embedding/embedding_weights:0Qinput_layer/sparse_input_layer/input_layer/C22_embedding/embedding_weights/AssignQinput_layer/sparse_input_layer/input_layer/C22_embedding/embedding_weights/read:02iinput_layer/sparse_input_layer/input_layer/C22_embedding/embedding_weights/Initializer/truncated_normal:08
б
Linput_layer/sparse_input_layer/input_layer/C23_embedding/embedding_weights:0Qinput_layer/sparse_input_layer/input_layer/C23_embedding/embedding_weights/AssignQinput_layer/sparse_input_layer/input_layer/C23_embedding/embedding_weights/read:02iinput_layer/sparse_input_layer/input_layer/C23_embedding/embedding_weights/Initializer/truncated_normal:08
б
Linput_layer/sparse_input_layer/input_layer/C24_embedding/embedding_weights:0Qinput_layer/sparse_input_layer/input_layer/C24_embedding/embedding_weights/AssignQinput_layer/sparse_input_layer/input_layer/C24_embedding/embedding_weights/read:02iinput_layer/sparse_input_layer/input_layer/C24_embedding/embedding_weights/Initializer/truncated_normal:08
б
Linput_layer/sparse_input_layer/input_layer/C25_embedding/embedding_weights:0Qinput_layer/sparse_input_layer/input_layer/C25_embedding/embedding_weights/AssignQinput_layer/sparse_input_layer/input_layer/C25_embedding/embedding_weights/read:02iinput_layer/sparse_input_layer/input_layer/C25_embedding/embedding_weights/Initializer/truncated_normal:08
б
Linput_layer/sparse_input_layer/input_layer/C26_embedding/embedding_weights:0Qinput_layer/sparse_input_layer/input_layer/C26_embedding/embedding_weights/AssignQinput_layer/sparse_input_layer/input_layer/C26_embedding/embedding_weights/read:02iinput_layer/sparse_input_layer/input_layer/C26_embedding/embedding_weights/Initializer/truncated_normal:08
Ё
Kinput_layer/sparse_input_layer/input_layer/C2_embedding/embedding_weights:0Pinput_layer/sparse_input_layer/input_layer/C2_embedding/embedding_weights/AssignPinput_layer/sparse_input_layer/input_layer/C2_embedding/embedding_weights/read:02hinput_layer/sparse_input_layer/input_layer/C2_embedding/embedding_weights/Initializer/truncated_normal:08
Ё
Kinput_layer/sparse_input_layer/input_layer/C3_embedding/embedding_weights:0Pinput_layer/sparse_input_layer/input_layer/C3_embedding/embedding_weights/AssignPinput_layer/sparse_input_layer/input_layer/C3_embedding/embedding_weights/read:02hinput_layer/sparse_input_layer/input_layer/C3_embedding/embedding_weights/Initializer/truncated_normal:08
Ё
Kinput_layer/sparse_input_layer/input_layer/C4_embedding/embedding_weights:0Pinput_layer/sparse_input_layer/input_layer/C4_embedding/embedding_weights/AssignPinput_layer/sparse_input_layer/input_layer/C4_embedding/embedding_weights/read:02hinput_layer/sparse_input_layer/input_layer/C4_embedding/embedding_weights/Initializer/truncated_normal:08
Ё
Kinput_layer/sparse_input_layer/input_layer/C5_embedding/embedding_weights:0Pinput_layer/sparse_input_layer/input_layer/C5_embedding/embedding_weights/AssignPinput_layer/sparse_input_layer/input_layer/C5_embedding/embedding_weights/read:02hinput_layer/sparse_input_layer/input_layer/C5_embedding/embedding_weights/Initializer/truncated_normal:08
Ё
Kinput_layer/sparse_input_layer/input_layer/C6_embedding/embedding_weights:0Pinput_layer/sparse_input_layer/input_layer/C6_embedding/embedding_weights/AssignPinput_layer/sparse_input_layer/input_layer/C6_embedding/embedding_weights/read:02hinput_layer/sparse_input_layer/input_layer/C6_embedding/embedding_weights/Initializer/truncated_normal:08
Ё
Kinput_layer/sparse_input_layer/input_layer/C7_embedding/embedding_weights:0Pinput_layer/sparse_input_layer/input_layer/C7_embedding/embedding_weights/AssignPinput_layer/sparse_input_layer/input_layer/C7_embedding/embedding_weights/read:02hinput_layer/sparse_input_layer/input_layer/C7_embedding/embedding_weights/Initializer/truncated_normal:08
Ё
Kinput_layer/sparse_input_layer/input_layer/C8_embedding/embedding_weights:0Pinput_layer/sparse_input_layer/input_layer/C8_embedding/embedding_weights/AssignPinput_layer/sparse_input_layer/input_layer/C8_embedding/embedding_weights/read:02hinput_layer/sparse_input_layer/input_layer/C8_embedding/embedding_weights/Initializer/truncated_normal:08
Ё
Kinput_layer/sparse_input_layer/input_layer/C9_embedding/embedding_weights:0Pinput_layer/sparse_input_layer/input_layer/C9_embedding/embedding_weights/AssignPinput_layer/sparse_input_layer/input_layer/C9_embedding/embedding_weights/read:02hinput_layer/sparse_input_layer/input_layer/C9_embedding/embedding_weights/Initializer/truncated_normal:08"д~
trainable_variablesћ~…~
б
Linput_layer/sparse_input_layer/input_layer/C10_embedding/embedding_weights:0Qinput_layer/sparse_input_layer/input_layer/C10_embedding/embedding_weights/AssignQinput_layer/sparse_input_layer/input_layer/C10_embedding/embedding_weights/read:02iinput_layer/sparse_input_layer/input_layer/C10_embedding/embedding_weights/Initializer/truncated_normal:08
б
Linput_layer/sparse_input_layer/input_layer/C11_embedding/embedding_weights:0Qinput_layer/sparse_input_layer/input_layer/C11_embedding/embedding_weights/AssignQinput_layer/sparse_input_layer/input_layer/C11_embedding/embedding_weights/read:02iinput_layer/sparse_input_layer/input_layer/C11_embedding/embedding_weights/Initializer/truncated_normal:08
б
Linput_layer/sparse_input_layer/input_layer/C12_embedding/embedding_weights:0Qinput_layer/sparse_input_layer/input_layer/C12_embedding/embedding_weights/AssignQinput_layer/sparse_input_layer/input_layer/C12_embedding/embedding_weights/read:02iinput_layer/sparse_input_layer/input_layer/C12_embedding/embedding_weights/Initializer/truncated_normal:08
б
Linput_layer/sparse_input_layer/input_layer/C13_embedding/embedding_weights:0Qinput_layer/sparse_input_layer/input_layer/C13_embedding/embedding_weights/AssignQinput_layer/sparse_input_layer/input_layer/C13_embedding/embedding_weights/read:02iinput_layer/sparse_input_layer/input_layer/C13_embedding/embedding_weights/Initializer/truncated_normal:08
б
Linput_layer/sparse_input_layer/input_layer/C14_embedding/embedding_weights:0Qinput_layer/sparse_input_layer/input_layer/C14_embedding/embedding_weights/AssignQinput_layer/sparse_input_layer/input_layer/C14_embedding/embedding_weights/read:02iinput_layer/sparse_input_layer/input_layer/C14_embedding/embedding_weights/Initializer/truncated_normal:08
б
Linput_layer/sparse_input_layer/input_layer/C15_embedding/embedding_weights:0Qinput_layer/sparse_input_layer/input_layer/C15_embedding/embedding_weights/AssignQinput_layer/sparse_input_layer/input_layer/C15_embedding/embedding_weights/read:02iinput_layer/sparse_input_layer/input_layer/C15_embedding/embedding_weights/Initializer/truncated_normal:08
б
Linput_layer/sparse_input_layer/input_layer/C16_embedding/embedding_weights:0Qinput_layer/sparse_input_layer/input_layer/C16_embedding/embedding_weights/AssignQinput_layer/sparse_input_layer/input_layer/C16_embedding/embedding_weights/read:02iinput_layer/sparse_input_layer/input_layer/C16_embedding/embedding_weights/Initializer/truncated_normal:08
б
Linput_layer/sparse_input_layer/input_layer/C17_embedding/embedding_weights:0Qinput_layer/sparse_input_layer/input_layer/C17_embedding/embedding_weights/AssignQinput_layer/sparse_input_layer/input_layer/C17_embedding/embedding_weights/read:02iinput_layer/sparse_input_layer/input_layer/C17_embedding/embedding_weights/Initializer/truncated_normal:08
б
Linput_layer/sparse_input_layer/input_layer/C18_embedding/embedding_weights:0Qinput_layer/sparse_input_layer/input_layer/C18_embedding/embedding_weights/AssignQinput_layer/sparse_input_layer/input_layer/C18_embedding/embedding_weights/read:02iinput_layer/sparse_input_layer/input_layer/C18_embedding/embedding_weights/Initializer/truncated_normal:08
б
Linput_layer/sparse_input_layer/input_layer/C19_embedding/embedding_weights:0Qinput_layer/sparse_input_layer/input_layer/C19_embedding/embedding_weights/AssignQinput_layer/sparse_input_layer/input_layer/C19_embedding/embedding_weights/read:02iinput_layer/sparse_input_layer/input_layer/C19_embedding/embedding_weights/Initializer/truncated_normal:08
Ё
Kinput_layer/sparse_input_layer/input_layer/C1_embedding/embedding_weights:0Pinput_layer/sparse_input_layer/input_layer/C1_embedding/embedding_weights/AssignPinput_layer/sparse_input_layer/input_layer/C1_embedding/embedding_weights/read:02hinput_layer/sparse_input_layer/input_layer/C1_embedding/embedding_weights/Initializer/truncated_normal:08
б
Linput_layer/sparse_input_layer/input_layer/C20_embedding/embedding_weights:0Qinput_layer/sparse_input_layer/input_layer/C20_embedding/embedding_weights/AssignQinput_layer/sparse_input_layer/input_layer/C20_embedding/embedding_weights/read:02iinput_layer/sparse_input_layer/input_layer/C20_embedding/embedding_weights/Initializer/truncated_normal:08
б
Linput_layer/sparse_input_layer/input_layer/C21_embedding/embedding_weights:0Qinput_layer/sparse_input_layer/input_layer/C21_embedding/embedding_weights/AssignQinput_layer/sparse_input_layer/input_layer/C21_embedding/embedding_weights/read:02iinput_layer/sparse_input_layer/input_layer/C21_embedding/embedding_weights/Initializer/truncated_normal:08
б
Linput_layer/sparse_input_layer/input_layer/C22_embedding/embedding_weights:0Qinput_layer/sparse_input_layer/input_layer/C22_embedding/embedding_weights/AssignQinput_layer/sparse_input_layer/input_layer/C22_embedding/embedding_weights/read:02iinput_layer/sparse_input_layer/input_layer/C22_embedding/embedding_weights/Initializer/truncated_normal:08
б
Linput_layer/sparse_input_layer/input_layer/C23_embedding/embedding_weights:0Qinput_layer/sparse_input_layer/input_layer/C23_embedding/embedding_weights/AssignQinput_layer/sparse_input_layer/input_layer/C23_embedding/embedding_weights/read:02iinput_layer/sparse_input_layer/input_layer/C23_embedding/embedding_weights/Initializer/truncated_normal:08
б
Linput_layer/sparse_input_layer/input_layer/C24_embedding/embedding_weights:0Qinput_layer/sparse_input_layer/input_layer/C24_embedding/embedding_weights/AssignQinput_layer/sparse_input_layer/input_layer/C24_embedding/embedding_weights/read:02iinput_layer/sparse_input_layer/input_layer/C24_embedding/embedding_weights/Initializer/truncated_normal:08
б
Linput_layer/sparse_input_layer/input_layer/C25_embedding/embedding_weights:0Qinput_layer/sparse_input_layer/input_layer/C25_embedding/embedding_weights/AssignQinput_layer/sparse_input_layer/input_layer/C25_embedding/embedding_weights/read:02iinput_layer/sparse_input_layer/input_layer/C25_embedding/embedding_weights/Initializer/truncated_normal:08
б
Linput_layer/sparse_input_layer/input_layer/C26_embedding/embedding_weights:0Qinput_layer/sparse_input_layer/input_layer/C26_embedding/embedding_weights/AssignQinput_layer/sparse_input_layer/input_layer/C26_embedding/embedding_weights/read:02iinput_layer/sparse_input_layer/input_layer/C26_embedding/embedding_weights/Initializer/truncated_normal:08
Ё
Kinput_layer/sparse_input_layer/input_layer/C2_embedding/embedding_weights:0Pinput_layer/sparse_input_layer/input_layer/C2_embedding/embedding_weights/AssignPinput_layer/sparse_input_layer/input_layer/C2_embedding/embedding_weights/read:02hinput_layer/sparse_input_layer/input_layer/C2_embedding/embedding_weights/Initializer/truncated_normal:08
Ё
Kinput_layer/sparse_input_layer/input_layer/C3_embedding/embedding_weights:0Pinput_layer/sparse_input_layer/input_layer/C3_embedding/embedding_weights/AssignPinput_layer/sparse_input_layer/input_layer/C3_embedding/embedding_weights/read:02hinput_layer/sparse_input_layer/input_layer/C3_embedding/embedding_weights/Initializer/truncated_normal:08
Ё
Kinput_layer/sparse_input_layer/input_layer/C4_embedding/embedding_weights:0Pinput_layer/sparse_input_layer/input_layer/C4_embedding/embedding_weights/AssignPinput_layer/sparse_input_layer/input_layer/C4_embedding/embedding_weights/read:02hinput_layer/sparse_input_layer/input_layer/C4_embedding/embedding_weights/Initializer/truncated_normal:08
Ё
Kinput_layer/sparse_input_layer/input_layer/C5_embedding/embedding_weights:0Pinput_layer/sparse_input_layer/input_layer/C5_embedding/embedding_weights/AssignPinput_layer/sparse_input_layer/input_layer/C5_embedding/embedding_weights/read:02hinput_layer/sparse_input_layer/input_layer/C5_embedding/embedding_weights/Initializer/truncated_normal:08
Ё
Kinput_layer/sparse_input_layer/input_layer/C6_embedding/embedding_weights:0Pinput_layer/sparse_input_layer/input_layer/C6_embedding/embedding_weights/AssignPinput_layer/sparse_input_layer/input_layer/C6_embedding/embedding_weights/read:02hinput_layer/sparse_input_layer/input_layer/C6_embedding/embedding_weights/Initializer/truncated_normal:08
Ё
Kinput_layer/sparse_input_layer/input_layer/C7_embedding/embedding_weights:0Pinput_layer/sparse_input_layer/input_layer/C7_embedding/embedding_weights/AssignPinput_layer/sparse_input_layer/input_layer/C7_embedding/embedding_weights/read:02hinput_layer/sparse_input_layer/input_layer/C7_embedding/embedding_weights/Initializer/truncated_normal:08
Ё
Kinput_layer/sparse_input_layer/input_layer/C8_embedding/embedding_weights:0Pinput_layer/sparse_input_layer/input_layer/C8_embedding/embedding_weights/AssignPinput_layer/sparse_input_layer/input_layer/C8_embedding/embedding_weights/read:02hinput_layer/sparse_input_layer/input_layer/C8_embedding/embedding_weights/Initializer/truncated_normal:08
Ё
Kinput_layer/sparse_input_layer/input_layer/C9_embedding/embedding_weights:0Pinput_layer/sparse_input_layer/input_layer/C9_embedding/embedding_weights/AssignPinput_layer/sparse_input_layer/input_layer/C9_embedding/embedding_weights/read:02hinput_layer/sparse_input_layer/input_layer/C9_embedding/embedding_weights/Initializer/truncated_normal:08
Ј
3mlp_bot_layer/mlp_bot_hiddenlayer_0/kernel/part_0:08mlp_bot_layer/mlp_bot_hiddenlayer_0/kernel/part_0/Assign8mlp_bot_layer/mlp_bot_hiddenlayer_0/kernel/part_0/read:0":
*mlp_bot_layer/mlp_bot_hiddenlayer_0/kernelА  "А2Nmlp_bot_layer/mlp_bot_hiddenlayer_0/kernel/part_0/Initializer/random_uniform:08
°
1mlp_bot_layer/mlp_bot_hiddenlayer_0/bias/part_0:06mlp_bot_layer/mlp_bot_hiddenlayer_0/bias/part_0/Assign6mlp_bot_layer/mlp_bot_hiddenlayer_0/bias/part_0/read:0"5
(mlp_bot_layer/mlp_bot_hiddenlayer_0/biasА "А2Cmlp_bot_layer/mlp_bot_hiddenlayer_0/bias/part_0/Initializer/zeros:08
Й
Fmlp_bot_layer/mlp_bot_hiddenlayer_0/batch_normalization/gamma/part_0:0Kmlp_bot_layer/mlp_bot_hiddenlayer_0/batch_normalization/gamma/part_0/AssignKmlp_bot_layer/mlp_bot_hiddenlayer_0/batch_normalization/gamma/part_0/read:0"J
=mlp_bot_layer/mlp_bot_hiddenlayer_0/batch_normalization/gammaА "А2Wmlp_bot_layer/mlp_bot_hiddenlayer_0/batch_normalization/gamma/part_0/Initializer/ones:08
Е
Emlp_bot_layer/mlp_bot_hiddenlayer_0/batch_normalization/beta/part_0:0Jmlp_bot_layer/mlp_bot_hiddenlayer_0/batch_normalization/beta/part_0/AssignJmlp_bot_layer/mlp_bot_hiddenlayer_0/batch_normalization/beta/part_0/read:0"I
<mlp_bot_layer/mlp_bot_hiddenlayer_0/batch_normalization/betaА "А2Wmlp_bot_layer/mlp_bot_hiddenlayer_0/batch_normalization/beta/part_0/Initializer/zeros:08
є
3mlp_bot_layer/mlp_bot_hiddenlayer_1/kernel/part_0:08mlp_bot_layer/mlp_bot_hiddenlayer_1/kernel/part_0/Assign8mlp_bot_layer/mlp_bot_hiddenlayer_1/kernel/part_0/read:0"<
*mlp_bot_layer/mlp_bot_hiddenlayer_1/kernelАА  "АА2Nmlp_bot_layer/mlp_bot_hiddenlayer_1/kernel/part_0/Initializer/random_uniform:08
°
1mlp_bot_layer/mlp_bot_hiddenlayer_1/bias/part_0:06mlp_bot_layer/mlp_bot_hiddenlayer_1/bias/part_0/Assign6mlp_bot_layer/mlp_bot_hiddenlayer_1/bias/part_0/read:0"5
(mlp_bot_layer/mlp_bot_hiddenlayer_1/biasА "А2Cmlp_bot_layer/mlp_bot_hiddenlayer_1/bias/part_0/Initializer/zeros:08
Й
Fmlp_bot_layer/mlp_bot_hiddenlayer_1/batch_normalization/gamma/part_0:0Kmlp_bot_layer/mlp_bot_hiddenlayer_1/batch_normalization/gamma/part_0/AssignKmlp_bot_layer/mlp_bot_hiddenlayer_1/batch_normalization/gamma/part_0/read:0"J
=mlp_bot_layer/mlp_bot_hiddenlayer_1/batch_normalization/gammaА "А2Wmlp_bot_layer/mlp_bot_hiddenlayer_1/batch_normalization/gamma/part_0/Initializer/ones:08
Е
Emlp_bot_layer/mlp_bot_hiddenlayer_1/batch_normalization/beta/part_0:0Jmlp_bot_layer/mlp_bot_hiddenlayer_1/batch_normalization/beta/part_0/AssignJmlp_bot_layer/mlp_bot_hiddenlayer_1/batch_normalization/beta/part_0/read:0"I
<mlp_bot_layer/mlp_bot_hiddenlayer_1/batch_normalization/betaА "А2Wmlp_bot_layer/mlp_bot_hiddenlayer_1/batch_normalization/beta/part_0/Initializer/zeros:08
Ј
3mlp_bot_layer/mlp_bot_hiddenlayer_2/kernel/part_0:08mlp_bot_layer/mlp_bot_hiddenlayer_2/kernel/part_0/Assign8mlp_bot_layer/mlp_bot_hiddenlayer_2/kernel/part_0/read:0":
*mlp_bot_layer/mlp_bot_hiddenlayer_2/kernelА@  "А@2Nmlp_bot_layer/mlp_bot_hiddenlayer_2/kernel/part_0/Initializer/random_uniform:08
Я
1mlp_bot_layer/mlp_bot_hiddenlayer_2/bias/part_0:06mlp_bot_layer/mlp_bot_hiddenlayer_2/bias/part_0/Assign6mlp_bot_layer/mlp_bot_hiddenlayer_2/bias/part_0/read:0"3
(mlp_bot_layer/mlp_bot_hiddenlayer_2/bias@ "@2Cmlp_bot_layer/mlp_bot_hiddenlayer_2/bias/part_0/Initializer/zeros:08
З
Fmlp_bot_layer/mlp_bot_hiddenlayer_2/batch_normalization/gamma/part_0:0Kmlp_bot_layer/mlp_bot_hiddenlayer_2/batch_normalization/gamma/part_0/AssignKmlp_bot_layer/mlp_bot_hiddenlayer_2/batch_normalization/gamma/part_0/read:0"H
=mlp_bot_layer/mlp_bot_hiddenlayer_2/batch_normalization/gamma@ "@2Wmlp_bot_layer/mlp_bot_hiddenlayer_2/batch_normalization/gamma/part_0/Initializer/ones:08
Г
Emlp_bot_layer/mlp_bot_hiddenlayer_2/batch_normalization/beta/part_0:0Jmlp_bot_layer/mlp_bot_hiddenlayer_2/batch_normalization/beta/part_0/AssignJmlp_bot_layer/mlp_bot_hiddenlayer_2/batch_normalization/beta/part_0/read:0"G
<mlp_bot_layer/mlp_bot_hiddenlayer_2/batch_normalization/beta@ "@2Wmlp_bot_layer/mlp_bot_hiddenlayer_2/batch_normalization/beta/part_0/Initializer/zeros:08
µ
3mlp_bot_layer/mlp_bot_hiddenlayer_3/kernel/part_0:08mlp_bot_layer/mlp_bot_hiddenlayer_3/kernel/part_0/Assign8mlp_bot_layer/mlp_bot_hiddenlayer_3/kernel/part_0/read:0"8
*mlp_bot_layer/mlp_bot_hiddenlayer_3/kernel@  "@2Nmlp_bot_layer/mlp_bot_hiddenlayer_3/kernel/part_0/Initializer/random_uniform:08
Я
1mlp_bot_layer/mlp_bot_hiddenlayer_3/bias/part_0:06mlp_bot_layer/mlp_bot_hiddenlayer_3/bias/part_0/Assign6mlp_bot_layer/mlp_bot_hiddenlayer_3/bias/part_0/read:0"3
(mlp_bot_layer/mlp_bot_hiddenlayer_3/bias "2Cmlp_bot_layer/mlp_bot_hiddenlayer_3/bias/part_0/Initializer/zeros:08
З
Fmlp_bot_layer/mlp_bot_hiddenlayer_3/batch_normalization/gamma/part_0:0Kmlp_bot_layer/mlp_bot_hiddenlayer_3/batch_normalization/gamma/part_0/AssignKmlp_bot_layer/mlp_bot_hiddenlayer_3/batch_normalization/gamma/part_0/read:0"H
=mlp_bot_layer/mlp_bot_hiddenlayer_3/batch_normalization/gamma "2Wmlp_bot_layer/mlp_bot_hiddenlayer_3/batch_normalization/gamma/part_0/Initializer/ones:08
Г
Emlp_bot_layer/mlp_bot_hiddenlayer_3/batch_normalization/beta/part_0:0Jmlp_bot_layer/mlp_bot_hiddenlayer_3/batch_normalization/beta/part_0/AssignJmlp_bot_layer/mlp_bot_hiddenlayer_3/batch_normalization/beta/part_0/read:0"G
<mlp_bot_layer/mlp_bot_hiddenlayer_3/batch_normalization/beta "2Wmlp_bot_layer/mlp_bot_hiddenlayer_3/batch_normalization/beta/part_0/Initializer/zeros:08
є
3mlp_top_layer/mlp_top_hiddenlayer_0/kernel/part_0:08mlp_top_layer/mlp_top_hiddenlayer_0/kernel/part_0/Assign8mlp_top_layer/mlp_top_hiddenlayer_0/kernel/part_0/read:0"<
*mlp_top_layer/mlp_top_hiddenlayer_0/kernel∞А  "∞А2Nmlp_top_layer/mlp_top_hiddenlayer_0/kernel/part_0/Initializer/random_uniform:08
°
1mlp_top_layer/mlp_top_hiddenlayer_0/bias/part_0:06mlp_top_layer/mlp_top_hiddenlayer_0/bias/part_0/Assign6mlp_top_layer/mlp_top_hiddenlayer_0/bias/part_0/read:0"5
(mlp_top_layer/mlp_top_hiddenlayer_0/biasА "А2Cmlp_top_layer/mlp_top_hiddenlayer_0/bias/part_0/Initializer/zeros:08
є
3mlp_top_layer/mlp_top_hiddenlayer_1/kernel/part_0:08mlp_top_layer/mlp_top_hiddenlayer_1/kernel/part_0/Assign8mlp_top_layer/mlp_top_hiddenlayer_1/kernel/part_0/read:0"<
*mlp_top_layer/mlp_top_hiddenlayer_1/kernel∞А  "∞А2Nmlp_top_layer/mlp_top_hiddenlayer_1/kernel/part_0/Initializer/random_uniform:08
°
1mlp_top_layer/mlp_top_hiddenlayer_1/bias/part_0:06mlp_top_layer/mlp_top_hiddenlayer_1/bias/part_0/Assign6mlp_top_layer/mlp_top_hiddenlayer_1/bias/part_0/read:0"5
(mlp_top_layer/mlp_top_hiddenlayer_1/biasА "А2Cmlp_top_layer/mlp_top_hiddenlayer_1/bias/part_0/Initializer/zeros:08
k
logits/kernel:0logits/kernel/Assignlogits/kernel/read:02*logits/kernel/Initializer/random_uniform:08
Z
logits/bias:0logits/bias/Assignlogits/bias/read:02logits/bias/Initializer/zeros:08"Фt
cond_contextГtАt
р	
@mlp_bot_layer/mlp_bot_hiddenlayer_0/zero_fraction/cond/cond_text@mlp_bot_layer/mlp_bot_hiddenlayer_0/zero_fraction/cond/pred_id:0Amlp_bot_layer/mlp_bot_hiddenlayer_0/zero_fraction/cond/switch_t:0 *§
Imlp_bot_layer/mlp_bot_hiddenlayer_0/batch_normalization/batchnorm/add_1:0
=mlp_bot_layer/mlp_bot_hiddenlayer_0/zero_fraction/cond/Cast:0
Kmlp_bot_layer/mlp_bot_hiddenlayer_0/zero_fraction/cond/count_nonzero/Cast:0
Lmlp_bot_layer/mlp_bot_hiddenlayer_0/zero_fraction/cond/count_nonzero/Const:0
Vmlp_bot_layer/mlp_bot_hiddenlayer_0/zero_fraction/cond/count_nonzero/NotEqual/Switch:1
Omlp_bot_layer/mlp_bot_hiddenlayer_0/zero_fraction/cond/count_nonzero/NotEqual:0
Tmlp_bot_layer/mlp_bot_hiddenlayer_0/zero_fraction/cond/count_nonzero/nonzero_count:0
Lmlp_bot_layer/mlp_bot_hiddenlayer_0/zero_fraction/cond/count_nonzero/zeros:0
@mlp_bot_layer/mlp_bot_hiddenlayer_0/zero_fraction/cond/pred_id:0
Amlp_bot_layer/mlp_bot_hiddenlayer_0/zero_fraction/cond/switch_t:0Д
@mlp_bot_layer/mlp_bot_hiddenlayer_0/zero_fraction/cond/pred_id:0@mlp_bot_layer/mlp_bot_hiddenlayer_0/zero_fraction/cond/pred_id:0£
Imlp_bot_layer/mlp_bot_hiddenlayer_0/batch_normalization/batchnorm/add_1:0Vmlp_bot_layer/mlp_bot_hiddenlayer_0/zero_fraction/cond/count_nonzero/NotEqual/Switch:1
њ	
Bmlp_bot_layer/mlp_bot_hiddenlayer_0/zero_fraction/cond/cond_text_1@mlp_bot_layer/mlp_bot_hiddenlayer_0/zero_fraction/cond/pred_id:0Amlp_bot_layer/mlp_bot_hiddenlayer_0/zero_fraction/cond/switch_f:0*у
Imlp_bot_layer/mlp_bot_hiddenlayer_0/batch_normalization/batchnorm/add_1:0
Mmlp_bot_layer/mlp_bot_hiddenlayer_0/zero_fraction/cond/count_nonzero_1/Cast:0
Nmlp_bot_layer/mlp_bot_hiddenlayer_0/zero_fraction/cond/count_nonzero_1/Const:0
Xmlp_bot_layer/mlp_bot_hiddenlayer_0/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0
Qmlp_bot_layer/mlp_bot_hiddenlayer_0/zero_fraction/cond/count_nonzero_1/NotEqual:0
Vmlp_bot_layer/mlp_bot_hiddenlayer_0/zero_fraction/cond/count_nonzero_1/nonzero_count:0
Nmlp_bot_layer/mlp_bot_hiddenlayer_0/zero_fraction/cond/count_nonzero_1/zeros:0
@mlp_bot_layer/mlp_bot_hiddenlayer_0/zero_fraction/cond/pred_id:0
Amlp_bot_layer/mlp_bot_hiddenlayer_0/zero_fraction/cond/switch_f:0Д
@mlp_bot_layer/mlp_bot_hiddenlayer_0/zero_fraction/cond/pred_id:0@mlp_bot_layer/mlp_bot_hiddenlayer_0/zero_fraction/cond/pred_id:0•
Imlp_bot_layer/mlp_bot_hiddenlayer_0/batch_normalization/batchnorm/add_1:0Xmlp_bot_layer/mlp_bot_hiddenlayer_0/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0
р	
@mlp_bot_layer/mlp_bot_hiddenlayer_1/zero_fraction/cond/cond_text@mlp_bot_layer/mlp_bot_hiddenlayer_1/zero_fraction/cond/pred_id:0Amlp_bot_layer/mlp_bot_hiddenlayer_1/zero_fraction/cond/switch_t:0 *§
Imlp_bot_layer/mlp_bot_hiddenlayer_1/batch_normalization/batchnorm/add_1:0
=mlp_bot_layer/mlp_bot_hiddenlayer_1/zero_fraction/cond/Cast:0
Kmlp_bot_layer/mlp_bot_hiddenlayer_1/zero_fraction/cond/count_nonzero/Cast:0
Lmlp_bot_layer/mlp_bot_hiddenlayer_1/zero_fraction/cond/count_nonzero/Const:0
Vmlp_bot_layer/mlp_bot_hiddenlayer_1/zero_fraction/cond/count_nonzero/NotEqual/Switch:1
Omlp_bot_layer/mlp_bot_hiddenlayer_1/zero_fraction/cond/count_nonzero/NotEqual:0
Tmlp_bot_layer/mlp_bot_hiddenlayer_1/zero_fraction/cond/count_nonzero/nonzero_count:0
Lmlp_bot_layer/mlp_bot_hiddenlayer_1/zero_fraction/cond/count_nonzero/zeros:0
@mlp_bot_layer/mlp_bot_hiddenlayer_1/zero_fraction/cond/pred_id:0
Amlp_bot_layer/mlp_bot_hiddenlayer_1/zero_fraction/cond/switch_t:0Д
@mlp_bot_layer/mlp_bot_hiddenlayer_1/zero_fraction/cond/pred_id:0@mlp_bot_layer/mlp_bot_hiddenlayer_1/zero_fraction/cond/pred_id:0£
Imlp_bot_layer/mlp_bot_hiddenlayer_1/batch_normalization/batchnorm/add_1:0Vmlp_bot_layer/mlp_bot_hiddenlayer_1/zero_fraction/cond/count_nonzero/NotEqual/Switch:1
њ	
Bmlp_bot_layer/mlp_bot_hiddenlayer_1/zero_fraction/cond/cond_text_1@mlp_bot_layer/mlp_bot_hiddenlayer_1/zero_fraction/cond/pred_id:0Amlp_bot_layer/mlp_bot_hiddenlayer_1/zero_fraction/cond/switch_f:0*у
Imlp_bot_layer/mlp_bot_hiddenlayer_1/batch_normalization/batchnorm/add_1:0
Mmlp_bot_layer/mlp_bot_hiddenlayer_1/zero_fraction/cond/count_nonzero_1/Cast:0
Nmlp_bot_layer/mlp_bot_hiddenlayer_1/zero_fraction/cond/count_nonzero_1/Const:0
Xmlp_bot_layer/mlp_bot_hiddenlayer_1/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0
Qmlp_bot_layer/mlp_bot_hiddenlayer_1/zero_fraction/cond/count_nonzero_1/NotEqual:0
Vmlp_bot_layer/mlp_bot_hiddenlayer_1/zero_fraction/cond/count_nonzero_1/nonzero_count:0
Nmlp_bot_layer/mlp_bot_hiddenlayer_1/zero_fraction/cond/count_nonzero_1/zeros:0
@mlp_bot_layer/mlp_bot_hiddenlayer_1/zero_fraction/cond/pred_id:0
Amlp_bot_layer/mlp_bot_hiddenlayer_1/zero_fraction/cond/switch_f:0Д
@mlp_bot_layer/mlp_bot_hiddenlayer_1/zero_fraction/cond/pred_id:0@mlp_bot_layer/mlp_bot_hiddenlayer_1/zero_fraction/cond/pred_id:0•
Imlp_bot_layer/mlp_bot_hiddenlayer_1/batch_normalization/batchnorm/add_1:0Xmlp_bot_layer/mlp_bot_hiddenlayer_1/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0
р	
@mlp_bot_layer/mlp_bot_hiddenlayer_2/zero_fraction/cond/cond_text@mlp_bot_layer/mlp_bot_hiddenlayer_2/zero_fraction/cond/pred_id:0Amlp_bot_layer/mlp_bot_hiddenlayer_2/zero_fraction/cond/switch_t:0 *§
Imlp_bot_layer/mlp_bot_hiddenlayer_2/batch_normalization/batchnorm/add_1:0
=mlp_bot_layer/mlp_bot_hiddenlayer_2/zero_fraction/cond/Cast:0
Kmlp_bot_layer/mlp_bot_hiddenlayer_2/zero_fraction/cond/count_nonzero/Cast:0
Lmlp_bot_layer/mlp_bot_hiddenlayer_2/zero_fraction/cond/count_nonzero/Const:0
Vmlp_bot_layer/mlp_bot_hiddenlayer_2/zero_fraction/cond/count_nonzero/NotEqual/Switch:1
Omlp_bot_layer/mlp_bot_hiddenlayer_2/zero_fraction/cond/count_nonzero/NotEqual:0
Tmlp_bot_layer/mlp_bot_hiddenlayer_2/zero_fraction/cond/count_nonzero/nonzero_count:0
Lmlp_bot_layer/mlp_bot_hiddenlayer_2/zero_fraction/cond/count_nonzero/zeros:0
@mlp_bot_layer/mlp_bot_hiddenlayer_2/zero_fraction/cond/pred_id:0
Amlp_bot_layer/mlp_bot_hiddenlayer_2/zero_fraction/cond/switch_t:0£
Imlp_bot_layer/mlp_bot_hiddenlayer_2/batch_normalization/batchnorm/add_1:0Vmlp_bot_layer/mlp_bot_hiddenlayer_2/zero_fraction/cond/count_nonzero/NotEqual/Switch:1Д
@mlp_bot_layer/mlp_bot_hiddenlayer_2/zero_fraction/cond/pred_id:0@mlp_bot_layer/mlp_bot_hiddenlayer_2/zero_fraction/cond/pred_id:0
њ	
Bmlp_bot_layer/mlp_bot_hiddenlayer_2/zero_fraction/cond/cond_text_1@mlp_bot_layer/mlp_bot_hiddenlayer_2/zero_fraction/cond/pred_id:0Amlp_bot_layer/mlp_bot_hiddenlayer_2/zero_fraction/cond/switch_f:0*у
Imlp_bot_layer/mlp_bot_hiddenlayer_2/batch_normalization/batchnorm/add_1:0
Mmlp_bot_layer/mlp_bot_hiddenlayer_2/zero_fraction/cond/count_nonzero_1/Cast:0
Nmlp_bot_layer/mlp_bot_hiddenlayer_2/zero_fraction/cond/count_nonzero_1/Const:0
Xmlp_bot_layer/mlp_bot_hiddenlayer_2/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0
Qmlp_bot_layer/mlp_bot_hiddenlayer_2/zero_fraction/cond/count_nonzero_1/NotEqual:0
Vmlp_bot_layer/mlp_bot_hiddenlayer_2/zero_fraction/cond/count_nonzero_1/nonzero_count:0
Nmlp_bot_layer/mlp_bot_hiddenlayer_2/zero_fraction/cond/count_nonzero_1/zeros:0
@mlp_bot_layer/mlp_bot_hiddenlayer_2/zero_fraction/cond/pred_id:0
Amlp_bot_layer/mlp_bot_hiddenlayer_2/zero_fraction/cond/switch_f:0•
Imlp_bot_layer/mlp_bot_hiddenlayer_2/batch_normalization/batchnorm/add_1:0Xmlp_bot_layer/mlp_bot_hiddenlayer_2/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0Д
@mlp_bot_layer/mlp_bot_hiddenlayer_2/zero_fraction/cond/pred_id:0@mlp_bot_layer/mlp_bot_hiddenlayer_2/zero_fraction/cond/pred_id:0
р	
@mlp_bot_layer/mlp_bot_hiddenlayer_3/zero_fraction/cond/cond_text@mlp_bot_layer/mlp_bot_hiddenlayer_3/zero_fraction/cond/pred_id:0Amlp_bot_layer/mlp_bot_hiddenlayer_3/zero_fraction/cond/switch_t:0 *§
Imlp_bot_layer/mlp_bot_hiddenlayer_3/batch_normalization/batchnorm/add_1:0
=mlp_bot_layer/mlp_bot_hiddenlayer_3/zero_fraction/cond/Cast:0
Kmlp_bot_layer/mlp_bot_hiddenlayer_3/zero_fraction/cond/count_nonzero/Cast:0
Lmlp_bot_layer/mlp_bot_hiddenlayer_3/zero_fraction/cond/count_nonzero/Const:0
Vmlp_bot_layer/mlp_bot_hiddenlayer_3/zero_fraction/cond/count_nonzero/NotEqual/Switch:1
Omlp_bot_layer/mlp_bot_hiddenlayer_3/zero_fraction/cond/count_nonzero/NotEqual:0
Tmlp_bot_layer/mlp_bot_hiddenlayer_3/zero_fraction/cond/count_nonzero/nonzero_count:0
Lmlp_bot_layer/mlp_bot_hiddenlayer_3/zero_fraction/cond/count_nonzero/zeros:0
@mlp_bot_layer/mlp_bot_hiddenlayer_3/zero_fraction/cond/pred_id:0
Amlp_bot_layer/mlp_bot_hiddenlayer_3/zero_fraction/cond/switch_t:0£
Imlp_bot_layer/mlp_bot_hiddenlayer_3/batch_normalization/batchnorm/add_1:0Vmlp_bot_layer/mlp_bot_hiddenlayer_3/zero_fraction/cond/count_nonzero/NotEqual/Switch:1Д
@mlp_bot_layer/mlp_bot_hiddenlayer_3/zero_fraction/cond/pred_id:0@mlp_bot_layer/mlp_bot_hiddenlayer_3/zero_fraction/cond/pred_id:0
њ	
Bmlp_bot_layer/mlp_bot_hiddenlayer_3/zero_fraction/cond/cond_text_1@mlp_bot_layer/mlp_bot_hiddenlayer_3/zero_fraction/cond/pred_id:0Amlp_bot_layer/mlp_bot_hiddenlayer_3/zero_fraction/cond/switch_f:0*у
Imlp_bot_layer/mlp_bot_hiddenlayer_3/batch_normalization/batchnorm/add_1:0
Mmlp_bot_layer/mlp_bot_hiddenlayer_3/zero_fraction/cond/count_nonzero_1/Cast:0
Nmlp_bot_layer/mlp_bot_hiddenlayer_3/zero_fraction/cond/count_nonzero_1/Const:0
Xmlp_bot_layer/mlp_bot_hiddenlayer_3/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0
Qmlp_bot_layer/mlp_bot_hiddenlayer_3/zero_fraction/cond/count_nonzero_1/NotEqual:0
Vmlp_bot_layer/mlp_bot_hiddenlayer_3/zero_fraction/cond/count_nonzero_1/nonzero_count:0
Nmlp_bot_layer/mlp_bot_hiddenlayer_3/zero_fraction/cond/count_nonzero_1/zeros:0
@mlp_bot_layer/mlp_bot_hiddenlayer_3/zero_fraction/cond/pred_id:0
Amlp_bot_layer/mlp_bot_hiddenlayer_3/zero_fraction/cond/switch_f:0Д
@mlp_bot_layer/mlp_bot_hiddenlayer_3/zero_fraction/cond/pred_id:0@mlp_bot_layer/mlp_bot_hiddenlayer_3/zero_fraction/cond/pred_id:0•
Imlp_bot_layer/mlp_bot_hiddenlayer_3/batch_normalization/batchnorm/add_1:0Xmlp_bot_layer/mlp_bot_hiddenlayer_3/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0
ж
*mlp_top_layer/zero_fraction/cond/cond_text*mlp_top_layer/zero_fraction/cond/pred_id:0+mlp_top_layer/zero_fraction/cond/switch_t:0 *№
*mlp_top_layer/mlp_top_hiddenlayer_0/Relu:0
'mlp_top_layer/zero_fraction/cond/Cast:0
5mlp_top_layer/zero_fraction/cond/count_nonzero/Cast:0
6mlp_top_layer/zero_fraction/cond/count_nonzero/Const:0
@mlp_top_layer/zero_fraction/cond/count_nonzero/NotEqual/Switch:1
9mlp_top_layer/zero_fraction/cond/count_nonzero/NotEqual:0
>mlp_top_layer/zero_fraction/cond/count_nonzero/nonzero_count:0
6mlp_top_layer/zero_fraction/cond/count_nonzero/zeros:0
*mlp_top_layer/zero_fraction/cond/pred_id:0
+mlp_top_layer/zero_fraction/cond/switch_t:0X
*mlp_top_layer/zero_fraction/cond/pred_id:0*mlp_top_layer/zero_fraction/cond/pred_id:0n
*mlp_top_layer/mlp_top_hiddenlayer_0/Relu:0@mlp_top_layer/zero_fraction/cond/count_nonzero/NotEqual/Switch:1
Ћ
,mlp_top_layer/zero_fraction/cond/cond_text_1*mlp_top_layer/zero_fraction/cond/pred_id:0+mlp_top_layer/zero_fraction/cond/switch_f:0*Ѕ
*mlp_top_layer/mlp_top_hiddenlayer_0/Relu:0
7mlp_top_layer/zero_fraction/cond/count_nonzero_1/Cast:0
8mlp_top_layer/zero_fraction/cond/count_nonzero_1/Const:0
Bmlp_top_layer/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0
;mlp_top_layer/zero_fraction/cond/count_nonzero_1/NotEqual:0
@mlp_top_layer/zero_fraction/cond/count_nonzero_1/nonzero_count:0
8mlp_top_layer/zero_fraction/cond/count_nonzero_1/zeros:0
*mlp_top_layer/zero_fraction/cond/pred_id:0
+mlp_top_layer/zero_fraction/cond/switch_f:0p
*mlp_top_layer/mlp_top_hiddenlayer_0/Relu:0Bmlp_top_layer/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0X
*mlp_top_layer/zero_fraction/cond/pred_id:0*mlp_top_layer/zero_fraction/cond/pred_id:0
Д
,mlp_top_layer/zero_fraction_1/cond/cond_text,mlp_top_layer/zero_fraction_1/cond/pred_id:0-mlp_top_layer/zero_fraction_1/cond/switch_t:0 *ф
*mlp_top_layer/mlp_top_hiddenlayer_1/Relu:0
)mlp_top_layer/zero_fraction_1/cond/Cast:0
7mlp_top_layer/zero_fraction_1/cond/count_nonzero/Cast:0
8mlp_top_layer/zero_fraction_1/cond/count_nonzero/Const:0
Bmlp_top_layer/zero_fraction_1/cond/count_nonzero/NotEqual/Switch:1
;mlp_top_layer/zero_fraction_1/cond/count_nonzero/NotEqual:0
@mlp_top_layer/zero_fraction_1/cond/count_nonzero/nonzero_count:0
8mlp_top_layer/zero_fraction_1/cond/count_nonzero/zeros:0
,mlp_top_layer/zero_fraction_1/cond/pred_id:0
-mlp_top_layer/zero_fraction_1/cond/switch_t:0\
,mlp_top_layer/zero_fraction_1/cond/pred_id:0,mlp_top_layer/zero_fraction_1/cond/pred_id:0p
*mlp_top_layer/mlp_top_hiddenlayer_1/Relu:0Bmlp_top_layer/zero_fraction_1/cond/count_nonzero/NotEqual/Switch:1
з
.mlp_top_layer/zero_fraction_1/cond/cond_text_1,mlp_top_layer/zero_fraction_1/cond/pred_id:0-mlp_top_layer/zero_fraction_1/cond/switch_f:0*„
*mlp_top_layer/mlp_top_hiddenlayer_1/Relu:0
9mlp_top_layer/zero_fraction_1/cond/count_nonzero_1/Cast:0
:mlp_top_layer/zero_fraction_1/cond/count_nonzero_1/Const:0
Dmlp_top_layer/zero_fraction_1/cond/count_nonzero_1/NotEqual/Switch:0
=mlp_top_layer/zero_fraction_1/cond/count_nonzero_1/NotEqual:0
Bmlp_top_layer/zero_fraction_1/cond/count_nonzero_1/nonzero_count:0
:mlp_top_layer/zero_fraction_1/cond/count_nonzero_1/zeros:0
,mlp_top_layer/zero_fraction_1/cond/pred_id:0
-mlp_top_layer/zero_fraction_1/cond/switch_f:0\
,mlp_top_layer/zero_fraction_1/cond/pred_id:0,mlp_top_layer/zero_fraction_1/cond/pred_id:0r
*mlp_top_layer/mlp_top_hiddenlayer_1/Relu:0Dmlp_top_layer/zero_fraction_1/cond/count_nonzero_1/NotEqual/Switch:0
…
#logits/zero_fraction/cond/cond_text#logits/zero_fraction/cond/pred_id:0$logits/zero_fraction/cond/switch_t:0 *‘
logits/Sigmoid:0
 logits/zero_fraction/cond/Cast:0
.logits/zero_fraction/cond/count_nonzero/Cast:0
/logits/zero_fraction/cond/count_nonzero/Const:0
9logits/zero_fraction/cond/count_nonzero/NotEqual/Switch:1
2logits/zero_fraction/cond/count_nonzero/NotEqual:0
7logits/zero_fraction/cond/count_nonzero/nonzero_count:0
/logits/zero_fraction/cond/count_nonzero/zeros:0
#logits/zero_fraction/cond/pred_id:0
$logits/zero_fraction/cond/switch_t:0M
logits/Sigmoid:09logits/zero_fraction/cond/count_nonzero/NotEqual/Switch:1J
#logits/zero_fraction/cond/pred_id:0#logits/zero_fraction/cond/pred_id:0
µ
%logits/zero_fraction/cond/cond_text_1#logits/zero_fraction/cond/pred_id:0$logits/zero_fraction/cond/switch_f:0*ј
logits/Sigmoid:0
0logits/zero_fraction/cond/count_nonzero_1/Cast:0
1logits/zero_fraction/cond/count_nonzero_1/Const:0
;logits/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0
4logits/zero_fraction/cond/count_nonzero_1/NotEqual:0
9logits/zero_fraction/cond/count_nonzero_1/nonzero_count:0
1logits/zero_fraction/cond/count_nonzero_1/zeros:0
#logits/zero_fraction/cond/pred_id:0
$logits/zero_fraction/cond/switch_f:0O
logits/Sigmoid:0;logits/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0J
#logits/zero_fraction/cond/pred_id:0#logits/zero_fraction/cond/pred_id:0*ш
serving_defaultд
*
C24#
Placeholder_36:0€€€€€€€€€
)
C1#
Placeholder_13:0€€€€€€€€€
*
C23#
Placeholder_35:0€€€€€€€€€
)
C8#
Placeholder_20:0€€€€€€€€€
*
C19#
Placeholder_31:0€€€€€€€€€
(
I9"
Placeholder_8:0€€€€€€€€€
*
C15#
Placeholder_27:0€€€€€€€€€
*
C13#
Placeholder_25:0€€€€€€€€€
&
I1 
Placeholder:0€€€€€€€€€
*
C26#
Placeholder_38:0€€€€€€€€€
)
C7#
Placeholder_19:0€€€€€€€€€
(
I2"
Placeholder_1:0€€€€€€€€€
(
I3"
Placeholder_2:0€€€€€€€€€
*
C18#
Placeholder_30:0€€€€€€€€€
*
C20#
Placeholder_32:0€€€€€€€€€
*
C14#
Placeholder_26:0€€€€€€€€€
)
C4#
Placeholder_16:0€€€€€€€€€
(
I5"
Placeholder_4:0€€€€€€€€€
*
I11#
Placeholder_10:0€€€€€€€€€
*
C17#
Placeholder_29:0€€€€€€€€€
(
I4"
Placeholder_3:0€€€€€€€€€
)
C5#
Placeholder_17:0€€€€€€€€€
*
C16#
Placeholder_28:0€€€€€€€€€
)
I10"
Placeholder_9:0€€€€€€€€€
)
C2#
Placeholder_14:0€€€€€€€€€
(
I6"
Placeholder_5:0€€€€€€€€€
)
C3#
Placeholder_15:0€€€€€€€€€
*
C22#
Placeholder_34:0€€€€€€€€€
*
C11#
Placeholder_23:0€€€€€€€€€
*
I13#
Placeholder_12:0€€€€€€€€€
*
C25#
Placeholder_37:0€€€€€€€€€
(
I7"
Placeholder_6:0€€€€€€€€€
*
C10#
Placeholder_22:0€€€€€€€€€
*
C21#
Placeholder_33:0€€€€€€€€€
*
C12#
Placeholder_24:0€€€€€€€€€
)
C9#
Placeholder_21:0€€€€€€€€€
*
I12#
Placeholder_11:0€€€€€€€€€
(
I8"
Placeholder_7:0€€€€€€€€€
)
C6#
Placeholder_18:0€€€€€€€€€0
score'
logits/Sigmoid:0€€€€€€€€€tensorflow/serving/predict