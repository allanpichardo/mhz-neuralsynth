??
??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
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
?
Conv2DBackpropInput
input_sizes
filter"T
out_backprop"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
.
Identity

input"T
output"T"	
Ttype
\
	LeakyRelu
features"T
activations"T"
alphafloat%??L>"
Ttype0:
2
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
?
Mul
x"T
y"T
z"T"
Ttype:
2	?

NoOp
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
@
ReadVariableOp
resource
value"dtype"
dtypetype?
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
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
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
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
?
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
-
Tanh
x"T
y"T"
Ttype:

2
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.6.02v2.6.0-rc2-32-g919f693420e8??
v
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*
shared_namedense/kernel
o
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel* 
_output_shapes
:
??*
dtype0
?
conv1d_transpose/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*(
shared_nameconv1d_transpose/kernel
?
+conv1d_transpose/kernel/Read/ReadVariableOpReadVariableOpconv1d_transpose/kernel*$
_output_shapes
:??*
dtype0
?
conv1d_transpose_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??**
shared_nameconv1d_transpose_1/kernel
?
-conv1d_transpose_1/kernel/Read/ReadVariableOpReadVariableOpconv1d_transpose_1/kernel*$
_output_shapes
:??*
dtype0
?
conv1d_transpose_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??**
shared_nameconv1d_transpose_2/kernel
?
-conv1d_transpose_2/kernel/Read/ReadVariableOpReadVariableOpconv1d_transpose_2/kernel*$
_output_shapes
:??*
dtype0
?
conv1d_transpose_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@?**
shared_nameconv1d_transpose_3/kernel
?
-conv1d_transpose_3/kernel/Read/ReadVariableOpReadVariableOpconv1d_transpose_3/kernel*#
_output_shapes
:@?*
dtype0
?
conv1d_transpose_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@**
shared_nameconv1d_transpose_4/kernel
?
-conv1d_transpose_4/kernel/Read/ReadVariableOpReadVariableOpconv1d_transpose_4/kernel*"
_output_shapes
:@*
dtype0

NoOpNoOp
?&
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?&
value?&B?& B?&
?
layer-0
layer_with_weights-0
layer-1
layer-2
layer-3
layer_with_weights-1
layer-4
layer-5
layer_with_weights-2
layer-6
layer-7
	layer_with_weights-3
	layer-8

layer-9
layer_with_weights-4
layer-10
layer-11
layer_with_weights-5
layer-12
regularization_losses
	variables
trainable_variables
	keras_api

signatures
 
^

kernel
regularization_losses
	variables
trainable_variables
	keras_api
R
regularization_losses
	variables
trainable_variables
	keras_api
R
regularization_losses
	variables
trainable_variables
	keras_api
^

 kernel
!regularization_losses
"	variables
#trainable_variables
$	keras_api
R
%regularization_losses
&	variables
'trainable_variables
(	keras_api
^

)kernel
*regularization_losses
+	variables
,trainable_variables
-	keras_api
R
.regularization_losses
/	variables
0trainable_variables
1	keras_api
^

2kernel
3regularization_losses
4	variables
5trainable_variables
6	keras_api
R
7regularization_losses
8	variables
9trainable_variables
:	keras_api
^

;kernel
<regularization_losses
=	variables
>trainable_variables
?	keras_api
R
@regularization_losses
A	variables
Btrainable_variables
C	keras_api
^

Dkernel
Eregularization_losses
F	variables
Gtrainable_variables
H	keras_api
 
*
0
 1
)2
23
;4
D5
*
0
 1
)2
23
;4
D5
?
Inon_trainable_variables

Jlayers
regularization_losses
Klayer_regularization_losses
Lmetrics
	variables
trainable_variables
Mlayer_metrics
 
XV
VARIABLE_VALUEdense/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
 

0

0
?
Nnon_trainable_variables

Olayers
regularization_losses
Player_regularization_losses
Qmetrics
	variables
trainable_variables
Rlayer_metrics
 
 
 
?
Snon_trainable_variables

Tlayers
regularization_losses
Ulayer_regularization_losses
Vmetrics
	variables
trainable_variables
Wlayer_metrics
 
 
 
?
Xnon_trainable_variables

Ylayers
regularization_losses
Zlayer_regularization_losses
[metrics
	variables
trainable_variables
\layer_metrics
ca
VARIABLE_VALUEconv1d_transpose/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
 

 0

 0
?
]non_trainable_variables

^layers
!regularization_losses
_layer_regularization_losses
`metrics
"	variables
#trainable_variables
alayer_metrics
 
 
 
?
bnon_trainable_variables

clayers
%regularization_losses
dlayer_regularization_losses
emetrics
&	variables
'trainable_variables
flayer_metrics
ec
VARIABLE_VALUEconv1d_transpose_1/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
 

)0

)0
?
gnon_trainable_variables

hlayers
*regularization_losses
ilayer_regularization_losses
jmetrics
+	variables
,trainable_variables
klayer_metrics
 
 
 
?
lnon_trainable_variables

mlayers
.regularization_losses
nlayer_regularization_losses
ometrics
/	variables
0trainable_variables
player_metrics
ec
VARIABLE_VALUEconv1d_transpose_2/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
 

20

20
?
qnon_trainable_variables

rlayers
3regularization_losses
slayer_regularization_losses
tmetrics
4	variables
5trainable_variables
ulayer_metrics
 
 
 
?
vnon_trainable_variables

wlayers
7regularization_losses
xlayer_regularization_losses
ymetrics
8	variables
9trainable_variables
zlayer_metrics
ec
VARIABLE_VALUEconv1d_transpose_3/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
 

;0

;0
?
{non_trainable_variables

|layers
<regularization_losses
}layer_regularization_losses
~metrics
=	variables
>trainable_variables
layer_metrics
 
 
 
?
?non_trainable_variables
?layers
@regularization_losses
 ?layer_regularization_losses
?metrics
A	variables
Btrainable_variables
?layer_metrics
ec
VARIABLE_VALUEconv1d_transpose_4/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
 

D0

D0
?
?non_trainable_variables
?layers
Eregularization_losses
 ?layer_regularization_losses
?metrics
F	variables
Gtrainable_variables
?layer_metrics
 
^
0
1
2
3
4
5
6
7
	8

9
10
11
12
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
z
serving_default_input_1Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1dense/kernelconv1d_transpose/kernelconv1d_transpose_1/kernelconv1d_transpose_2/kernelconv1d_transpose_3/kernelconv1d_transpose_4/kernel*
Tin
	2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:???????????*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? */
f*R(
&__inference_signature_wrapper_78011748
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename dense/kernel/Read/ReadVariableOp+conv1d_transpose/kernel/Read/ReadVariableOp-conv1d_transpose_1/kernel/Read/ReadVariableOp-conv1d_transpose_2/kernel/Read/ReadVariableOp-conv1d_transpose_3/kernel/Read/ReadVariableOp-conv1d_transpose_4/kernel/Read/ReadVariableOpConst*
Tin

2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? **
f%R#
!__inference__traced_save_78012701
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense/kernelconv1d_transpose/kernelconv1d_transpose_1/kernelconv1d_transpose_2/kernelconv1d_transpose_3/kernelconv1d_transpose_4/kernel*
Tin
	2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *-
f(R&
$__inference__traced_restore_78012729ǣ
?
?
5__inference_conv1d_transpose_3_layer_call_fn_78012562

inputs
unknown:@?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????? @*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_conv1d_transpose_3_layer_call_and_return_conditional_losses_780114482
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:?????????? @2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:???????????: 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
-
_output_shapes
:???????????
 
_user_specified_nameinputs
?/
?
P__inference_conv1d_transpose_4_layer_call_and_return_conditional_losses_78012609

inputsK
5conv1d_transpose_expanddims_1_readvariableop_resource:@
identity??,conv1d_transpose/ExpandDims_1/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulT
stack/2Const*
_output_shapes
: *
dtype0*
value	B :2	
stack/2w
stackPackstrided_slice:output:0mul:z:0stack/2:output:0*
N*
T0*
_output_shapes
:2
stack?
conv1d_transpose/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2!
conv1d_transpose/ExpandDims/dim?
conv1d_transpose/ExpandDims
ExpandDimsinputs(conv1d_transpose/ExpandDims/dim:output:0*
T0*8
_output_shapes&
$:"??????????????????@2
conv1d_transpose/ExpandDims?
,conv1d_transpose/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_transpose_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02.
,conv1d_transpose/ExpandDims_1/ReadVariableOp?
!conv1d_transpose/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!conv1d_transpose/ExpandDims_1/dim?
conv1d_transpose/ExpandDims_1
ExpandDims4conv1d_transpose/ExpandDims_1/ReadVariableOp:value:0*conv1d_transpose/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@2
conv1d_transpose/ExpandDims_1?
$conv1d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$conv1d_transpose/strided_slice/stack?
&conv1d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv1d_transpose/strided_slice/stack_1?
&conv1d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv1d_transpose/strided_slice/stack_2?
conv1d_transpose/strided_sliceStridedSlicestack:output:0-conv1d_transpose/strided_slice/stack:output:0/conv1d_transpose/strided_slice/stack_1:output:0/conv1d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2 
conv1d_transpose/strided_slice?
&conv1d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2(
&conv1d_transpose/strided_slice_1/stack?
(conv1d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(conv1d_transpose/strided_slice_1/stack_1?
(conv1d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv1d_transpose/strided_slice_1/stack_2?
 conv1d_transpose/strided_slice_1StridedSlicestack:output:0/conv1d_transpose/strided_slice_1/stack:output:01conv1d_transpose/strided_slice_1/stack_1:output:01conv1d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2"
 conv1d_transpose/strided_slice_1?
 conv1d_transpose/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 conv1d_transpose/concat/values_1~
conv1d_transpose/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d_transpose/concat/axis?
conv1d_transpose/concatConcatV2'conv1d_transpose/strided_slice:output:0)conv1d_transpose/concat/values_1:output:0)conv1d_transpose/strided_slice_1:output:0%conv1d_transpose/concat/axis:output:0*
N*
T0*
_output_shapes
:2
conv1d_transpose/concat?
conv1d_transposeConv2DBackpropInput conv1d_transpose/concat:output:0&conv1d_transpose/ExpandDims_1:output:0$conv1d_transpose/ExpandDims:output:0*
T0*8
_output_shapes&
$:"??????????????????*
paddingSAME*
strides
2
conv1d_transpose?
conv1d_transpose/SqueezeSqueezeconv1d_transpose:output:0*
T0*4
_output_shapes"
 :??????????????????*
squeeze_dims
2
conv1d_transpose/Squeezev
TanhTanh!conv1d_transpose/Squeeze:output:0*
T0*4
_output_shapes"
 :??????????????????2
Tanhp
IdentityIdentityTanh:y:0^NoOp*
T0*4
_output_shapes"
 :??????????????????2

Identity}
NoOpNoOp-^conv1d_transpose/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":??????????????????@: 2\
,conv1d_transpose/ExpandDims_1/ReadVariableOp,conv1d_transpose/ExpandDims_1/ReadVariableOp:\ X
4
_output_shapes"
 :??????????????????@
 
_user_specified_nameinputs
?.
?
N__inference_conv1d_transpose_layer_call_and_return_conditional_losses_78011310

inputsM
5conv1d_transpose_expanddims_1_readvariableop_resource:??
identity??,conv1d_transpose/ExpandDims_1/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulU
stack/2Const*
_output_shapes
: *
dtype0*
value
B :?2	
stack/2w
stackPackstrided_slice:output:0mul:z:0stack/2:output:0*
N*
T0*
_output_shapes
:2
stack?
conv1d_transpose/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2!
conv1d_transpose/ExpandDims/dim?
conv1d_transpose/ExpandDims
ExpandDimsinputs(conv1d_transpose/ExpandDims/dim:output:0*
T0*0
_output_shapes
:??????????2
conv1d_transpose/ExpandDims?
,conv1d_transpose/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_transpose_expanddims_1_readvariableop_resource*$
_output_shapes
:??*
dtype02.
,conv1d_transpose/ExpandDims_1/ReadVariableOp?
!conv1d_transpose/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!conv1d_transpose/ExpandDims_1/dim?
conv1d_transpose/ExpandDims_1
ExpandDims4conv1d_transpose/ExpandDims_1/ReadVariableOp:value:0*conv1d_transpose/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:??2
conv1d_transpose/ExpandDims_1?
$conv1d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$conv1d_transpose/strided_slice/stack?
&conv1d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv1d_transpose/strided_slice/stack_1?
&conv1d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv1d_transpose/strided_slice/stack_2?
conv1d_transpose/strided_sliceStridedSlicestack:output:0-conv1d_transpose/strided_slice/stack:output:0/conv1d_transpose/strided_slice/stack_1:output:0/conv1d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2 
conv1d_transpose/strided_slice?
&conv1d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2(
&conv1d_transpose/strided_slice_1/stack?
(conv1d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(conv1d_transpose/strided_slice_1/stack_1?
(conv1d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv1d_transpose/strided_slice_1/stack_2?
 conv1d_transpose/strided_slice_1StridedSlicestack:output:0/conv1d_transpose/strided_slice_1/stack:output:01conv1d_transpose/strided_slice_1/stack_1:output:01conv1d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2"
 conv1d_transpose/strided_slice_1?
 conv1d_transpose/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 conv1d_transpose/concat/values_1~
conv1d_transpose/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d_transpose/concat/axis?
conv1d_transpose/concatConcatV2'conv1d_transpose/strided_slice:output:0)conv1d_transpose/concat/values_1:output:0)conv1d_transpose/strided_slice_1:output:0%conv1d_transpose/concat/axis:output:0*
N*
T0*
_output_shapes
:2
conv1d_transpose/concat?
conv1d_transposeConv2DBackpropInput conv1d_transpose/concat:output:0&conv1d_transpose/ExpandDims_1:output:0$conv1d_transpose/ExpandDims:output:0*
T0*9
_output_shapes'
%:#???????????????????*
paddingSAME*
strides
2
conv1d_transpose?
conv1d_transpose/SqueezeSqueezeconv1d_transpose:output:0*
T0*,
_output_shapes
:?????????@?*
squeeze_dims
2
conv1d_transpose/Squeeze?
IdentityIdentity!conv1d_transpose/Squeeze:output:0^NoOp*
T0*,
_output_shapes
:?????????@?2

Identity}
NoOpNoOp-^conv1d_transpose/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*-
_input_shapes
:??????????: 2\
,conv1d_transpose/ExpandDims_1/ReadVariableOp,conv1d_transpose/ExpandDims_1/ReadVariableOp:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
C__inference_dense_layer_call_and_return_conditional_losses_78012153

inputs2
matmul_readvariableop_resource:
??
identity??MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpu
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*)
_output_shapes
:???????????2
MatMulm
IdentityIdentityMatMul:product:0^NoOp*
T0*)
_output_shapes
:???????????2

Identityf
NoOpNoOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?.
?
P__inference_conv1d_transpose_1_layer_call_and_return_conditional_losses_78011356

inputsM
5conv1d_transpose_expanddims_1_readvariableop_resource:??
identity??,conv1d_transpose/ExpandDims_1/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulU
stack/2Const*
_output_shapes
: *
dtype0*
value
B :?2	
stack/2w
stackPackstrided_slice:output:0mul:z:0stack/2:output:0*
N*
T0*
_output_shapes
:2
stack?
conv1d_transpose/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2!
conv1d_transpose/ExpandDims/dim?
conv1d_transpose/ExpandDims
ExpandDimsinputs(conv1d_transpose/ExpandDims/dim:output:0*
T0*0
_output_shapes
:?????????@?2
conv1d_transpose/ExpandDims?
,conv1d_transpose/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_transpose_expanddims_1_readvariableop_resource*$
_output_shapes
:??*
dtype02.
,conv1d_transpose/ExpandDims_1/ReadVariableOp?
!conv1d_transpose/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!conv1d_transpose/ExpandDims_1/dim?
conv1d_transpose/ExpandDims_1
ExpandDims4conv1d_transpose/ExpandDims_1/ReadVariableOp:value:0*conv1d_transpose/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:??2
conv1d_transpose/ExpandDims_1?
$conv1d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$conv1d_transpose/strided_slice/stack?
&conv1d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv1d_transpose/strided_slice/stack_1?
&conv1d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv1d_transpose/strided_slice/stack_2?
conv1d_transpose/strided_sliceStridedSlicestack:output:0-conv1d_transpose/strided_slice/stack:output:0/conv1d_transpose/strided_slice/stack_1:output:0/conv1d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2 
conv1d_transpose/strided_slice?
&conv1d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2(
&conv1d_transpose/strided_slice_1/stack?
(conv1d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(conv1d_transpose/strided_slice_1/stack_1?
(conv1d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv1d_transpose/strided_slice_1/stack_2?
 conv1d_transpose/strided_slice_1StridedSlicestack:output:0/conv1d_transpose/strided_slice_1/stack:output:01conv1d_transpose/strided_slice_1/stack_1:output:01conv1d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2"
 conv1d_transpose/strided_slice_1?
 conv1d_transpose/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 conv1d_transpose/concat/values_1~
conv1d_transpose/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d_transpose/concat/axis?
conv1d_transpose/concatConcatV2'conv1d_transpose/strided_slice:output:0)conv1d_transpose/concat/values_1:output:0)conv1d_transpose/strided_slice_1:output:0%conv1d_transpose/concat/axis:output:0*
N*
T0*
_output_shapes
:2
conv1d_transpose/concat?
conv1d_transposeConv2DBackpropInput conv1d_transpose/concat:output:0&conv1d_transpose/ExpandDims_1:output:0$conv1d_transpose/ExpandDims:output:0*
T0*9
_output_shapes'
%:#???????????????????*
paddingSAME*
strides
2
conv1d_transpose?
conv1d_transpose/SqueezeSqueezeconv1d_transpose:output:0*
T0*-
_output_shapes
:???????????*
squeeze_dims
2
conv1d_transpose/Squeeze?
IdentityIdentity!conv1d_transpose/Squeeze:output:0^NoOp*
T0*-
_output_shapes
:???????????2

Identity}
NoOpNoOp-^conv1d_transpose/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*-
_input_shapes
:?????????@?: 2\
,conv1d_transpose/ExpandDims_1/ReadVariableOp,conv1d_transpose/ExpandDims_1/ReadVariableOp:T P
,
_output_shapes
:?????????@?
 
_user_specified_nameinputs
?
L
0__inference_leaky_re_lu_3_layer_call_fn_78012476

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_leaky_re_lu_3_layer_call_and_return_conditional_losses_780114112
PartitionedCallr
IdentityIdentityPartitionedCall:output:0*
T0*-
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:???????????:U Q
-
_output_shapes
:???????????
 
_user_specified_nameinputs
?
J
.__inference_leaky_re_lu_layer_call_fn_78012170

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_leaky_re_lu_layer_call_and_return_conditional_losses_780112582
PartitionedCalln
IdentityIdentityPartitionedCall:output:0*
T0*)
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:???????????:Q M
)
_output_shapes
:???????????
 
_user_specified_nameinputs
?.
?
P__inference_conv1d_transpose_2_layer_call_and_return_conditional_losses_78012416

inputsM
5conv1d_transpose_expanddims_1_readvariableop_resource:??
identity??,conv1d_transpose/ExpandDims_1/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulU
stack/2Const*
_output_shapes
: *
dtype0*
value
B :?2	
stack/2w
stackPackstrided_slice:output:0mul:z:0stack/2:output:0*
N*
T0*
_output_shapes
:2
stack?
conv1d_transpose/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2!
conv1d_transpose/ExpandDims/dim?
conv1d_transpose/ExpandDims
ExpandDimsinputs(conv1d_transpose/ExpandDims/dim:output:0*
T0*9
_output_shapes'
%:#???????????????????2
conv1d_transpose/ExpandDims?
,conv1d_transpose/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_transpose_expanddims_1_readvariableop_resource*$
_output_shapes
:??*
dtype02.
,conv1d_transpose/ExpandDims_1/ReadVariableOp?
!conv1d_transpose/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!conv1d_transpose/ExpandDims_1/dim?
conv1d_transpose/ExpandDims_1
ExpandDims4conv1d_transpose/ExpandDims_1/ReadVariableOp:value:0*conv1d_transpose/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:??2
conv1d_transpose/ExpandDims_1?
$conv1d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$conv1d_transpose/strided_slice/stack?
&conv1d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv1d_transpose/strided_slice/stack_1?
&conv1d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv1d_transpose/strided_slice/stack_2?
conv1d_transpose/strided_sliceStridedSlicestack:output:0-conv1d_transpose/strided_slice/stack:output:0/conv1d_transpose/strided_slice/stack_1:output:0/conv1d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2 
conv1d_transpose/strided_slice?
&conv1d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2(
&conv1d_transpose/strided_slice_1/stack?
(conv1d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(conv1d_transpose/strided_slice_1/stack_1?
(conv1d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv1d_transpose/strided_slice_1/stack_2?
 conv1d_transpose/strided_slice_1StridedSlicestack:output:0/conv1d_transpose/strided_slice_1/stack:output:01conv1d_transpose/strided_slice_1/stack_1:output:01conv1d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2"
 conv1d_transpose/strided_slice_1?
 conv1d_transpose/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 conv1d_transpose/concat/values_1~
conv1d_transpose/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d_transpose/concat/axis?
conv1d_transpose/concatConcatV2'conv1d_transpose/strided_slice:output:0)conv1d_transpose/concat/values_1:output:0)conv1d_transpose/strided_slice_1:output:0%conv1d_transpose/concat/axis:output:0*
N*
T0*
_output_shapes
:2
conv1d_transpose/concat?
conv1d_transposeConv2DBackpropInput conv1d_transpose/concat:output:0&conv1d_transpose/ExpandDims_1:output:0$conv1d_transpose/ExpandDims:output:0*
T0*9
_output_shapes'
%:#???????????????????*
paddingSAME*
strides
2
conv1d_transpose?
conv1d_transpose/SqueezeSqueezeconv1d_transpose:output:0*
T0*5
_output_shapes#
!:???????????????????*
squeeze_dims
2
conv1d_transpose/Squeeze?
IdentityIdentity!conv1d_transpose/Squeeze:output:0^NoOp*
T0*5
_output_shapes#
!:???????????????????2

Identity}
NoOpNoOp-^conv1d_transpose/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:???????????????????: 2\
,conv1d_transpose/ExpandDims_1/ReadVariableOp,conv1d_transpose/ExpandDims_1/ReadVariableOp:] Y
5
_output_shapes#
!:???????????????????
 
_user_specified_nameinputs
?
g
K__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_78011365

inputs
identityj
	LeakyRelu	LeakyReluinputs*-
_output_shapes
:???????????*
alpha%???>2
	LeakyReluq
IdentityIdentityLeakyRelu:activations:0*
T0*-
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:???????????:U Q
-
_output_shapes
:???????????
 
_user_specified_nameinputs
?	
?
/__inference_WG_Generator_layer_call_fn_78011515
input_1
unknown:
??!
	unknown_0:??!
	unknown_1:??!
	unknown_2:?? 
	unknown_3:@?
	unknown_4:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:???????????*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_WG_Generator_layer_call_and_return_conditional_losses_780115002
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*-
_output_shapes
:???????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1
?/
?
P__inference_conv1d_transpose_4_layer_call_and_return_conditional_losses_78011186

inputsK
5conv1d_transpose_expanddims_1_readvariableop_resource:@
identity??,conv1d_transpose/ExpandDims_1/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulT
stack/2Const*
_output_shapes
: *
dtype0*
value	B :2	
stack/2w
stackPackstrided_slice:output:0mul:z:0stack/2:output:0*
N*
T0*
_output_shapes
:2
stack?
conv1d_transpose/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2!
conv1d_transpose/ExpandDims/dim?
conv1d_transpose/ExpandDims
ExpandDimsinputs(conv1d_transpose/ExpandDims/dim:output:0*
T0*8
_output_shapes&
$:"??????????????????@2
conv1d_transpose/ExpandDims?
,conv1d_transpose/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_transpose_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02.
,conv1d_transpose/ExpandDims_1/ReadVariableOp?
!conv1d_transpose/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!conv1d_transpose/ExpandDims_1/dim?
conv1d_transpose/ExpandDims_1
ExpandDims4conv1d_transpose/ExpandDims_1/ReadVariableOp:value:0*conv1d_transpose/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@2
conv1d_transpose/ExpandDims_1?
$conv1d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$conv1d_transpose/strided_slice/stack?
&conv1d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv1d_transpose/strided_slice/stack_1?
&conv1d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv1d_transpose/strided_slice/stack_2?
conv1d_transpose/strided_sliceStridedSlicestack:output:0-conv1d_transpose/strided_slice/stack:output:0/conv1d_transpose/strided_slice/stack_1:output:0/conv1d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2 
conv1d_transpose/strided_slice?
&conv1d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2(
&conv1d_transpose/strided_slice_1/stack?
(conv1d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(conv1d_transpose/strided_slice_1/stack_1?
(conv1d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv1d_transpose/strided_slice_1/stack_2?
 conv1d_transpose/strided_slice_1StridedSlicestack:output:0/conv1d_transpose/strided_slice_1/stack:output:01conv1d_transpose/strided_slice_1/stack_1:output:01conv1d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2"
 conv1d_transpose/strided_slice_1?
 conv1d_transpose/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 conv1d_transpose/concat/values_1~
conv1d_transpose/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d_transpose/concat/axis?
conv1d_transpose/concatConcatV2'conv1d_transpose/strided_slice:output:0)conv1d_transpose/concat/values_1:output:0)conv1d_transpose/strided_slice_1:output:0%conv1d_transpose/concat/axis:output:0*
N*
T0*
_output_shapes
:2
conv1d_transpose/concat?
conv1d_transposeConv2DBackpropInput conv1d_transpose/concat:output:0&conv1d_transpose/ExpandDims_1:output:0$conv1d_transpose/ExpandDims:output:0*
T0*8
_output_shapes&
$:"??????????????????*
paddingSAME*
strides
2
conv1d_transpose?
conv1d_transpose/SqueezeSqueezeconv1d_transpose:output:0*
T0*4
_output_shapes"
 :??????????????????*
squeeze_dims
2
conv1d_transpose/Squeezev
TanhTanh!conv1d_transpose/Squeeze:output:0*
T0*4
_output_shapes"
 :??????????????????2
Tanhp
IdentityIdentityTanh:y:0^NoOp*
T0*4
_output_shapes"
 :??????????????????2

Identity}
NoOpNoOp-^conv1d_transpose/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":??????????????????@: 2\
,conv1d_transpose/ExpandDims_1/ReadVariableOp,conv1d_transpose/ExpandDims_1/ReadVariableOp:\ X
4
_output_shapes"
 :??????????????????@
 
_user_specified_nameinputs
?
g
K__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_78012279

inputs
identityi
	LeakyRelu	LeakyReluinputs*,
_output_shapes
:?????????@?*
alpha%???>2
	LeakyRelup
IdentityIdentityLeakyRelu:activations:0*
T0*,
_output_shapes
:?????????@?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:?????????@?:T P
,
_output_shapes
:?????????@?
 
_user_specified_nameinputs
?6
?
J__inference_WG_Generator_layer_call_and_return_conditional_losses_78011641

inputs"
dense_78011616:
??1
conv1d_transpose_78011621:??3
conv1d_transpose_1_78011625:??3
conv1d_transpose_2_78011629:??2
conv1d_transpose_3_78011633:@?1
conv1d_transpose_4_78011637:@
identity??(conv1d_transpose/StatefulPartitionedCall?*conv1d_transpose_1/StatefulPartitionedCall?*conv1d_transpose_2/StatefulPartitionedCall?*conv1d_transpose_3/StatefulPartitionedCall?*conv1d_transpose_4/StatefulPartitionedCall?dense/StatefulPartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCallinputsdense_78011616*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_layer_call_and_return_conditional_losses_780112492
dense/StatefulPartitionedCall?
leaky_re_lu/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_leaky_re_lu_layer_call_and_return_conditional_losses_780112582
leaky_re_lu/PartitionedCall?
reshape/PartitionedCallPartitionedCall$leaky_re_lu/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_reshape_layer_call_and_return_conditional_losses_780112732
reshape/PartitionedCall?
(conv1d_transpose/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0conv1d_transpose_78011621*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????@?*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_conv1d_transpose_layer_call_and_return_conditional_losses_780113102*
(conv1d_transpose/StatefulPartitionedCall?
leaky_re_lu_1/PartitionedCallPartitionedCall1conv1d_transpose/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????@?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_780113192
leaky_re_lu_1/PartitionedCall?
*conv1d_transpose_1/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_1/PartitionedCall:output:0conv1d_transpose_1_78011625*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:???????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_conv1d_transpose_1_layer_call_and_return_conditional_losses_780113562,
*conv1d_transpose_1/StatefulPartitionedCall?
leaky_re_lu_2/PartitionedCallPartitionedCall3conv1d_transpose_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_780113652
leaky_re_lu_2/PartitionedCall?
*conv1d_transpose_2/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_2/PartitionedCall:output:0conv1d_transpose_2_78011629*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:???????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_conv1d_transpose_2_layer_call_and_return_conditional_losses_780114022,
*conv1d_transpose_2/StatefulPartitionedCall?
leaky_re_lu_3/PartitionedCallPartitionedCall3conv1d_transpose_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_leaky_re_lu_3_layer_call_and_return_conditional_losses_780114112
leaky_re_lu_3/PartitionedCall?
*conv1d_transpose_3/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_3/PartitionedCall:output:0conv1d_transpose_3_78011633*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????? @*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_conv1d_transpose_3_layer_call_and_return_conditional_losses_780114482,
*conv1d_transpose_3/StatefulPartitionedCall?
leaky_re_lu_4/PartitionedCallPartitionedCall3conv1d_transpose_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????? @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_leaky_re_lu_4_layer_call_and_return_conditional_losses_780114572
leaky_re_lu_4/PartitionedCall?
*conv1d_transpose_4/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_4/PartitionedCall:output:0conv1d_transpose_4_78011637*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:???????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_conv1d_transpose_4_layer_call_and_return_conditional_losses_780114952,
*conv1d_transpose_4/StatefulPartitionedCall?
IdentityIdentity3conv1d_transpose_4/StatefulPartitionedCall:output:0^NoOp*
T0*-
_output_shapes
:???????????2

Identity?
NoOpNoOp)^conv1d_transpose/StatefulPartitionedCall+^conv1d_transpose_1/StatefulPartitionedCall+^conv1d_transpose_2/StatefulPartitionedCall+^conv1d_transpose_3/StatefulPartitionedCall+^conv1d_transpose_4/StatefulPartitionedCall^dense/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : : : : : 2T
(conv1d_transpose/StatefulPartitionedCall(conv1d_transpose/StatefulPartitionedCall2X
*conv1d_transpose_1/StatefulPartitionedCall*conv1d_transpose_1/StatefulPartitionedCall2X
*conv1d_transpose_2/StatefulPartitionedCall*conv1d_transpose_2/StatefulPartitionedCall2X
*conv1d_transpose_3/StatefulPartitionedCall*conv1d_transpose_3/StatefulPartitionedCall2X
*conv1d_transpose_4/StatefulPartitionedCall*conv1d_transpose_4/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?.
?
P__inference_conv1d_transpose_3_layer_call_and_return_conditional_losses_78011097

inputsL
5conv1d_transpose_expanddims_1_readvariableop_resource:@?
identity??,conv1d_transpose/ExpandDims_1/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulT
stack/2Const*
_output_shapes
: *
dtype0*
value	B :@2	
stack/2w
stackPackstrided_slice:output:0mul:z:0stack/2:output:0*
N*
T0*
_output_shapes
:2
stack?
conv1d_transpose/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2!
conv1d_transpose/ExpandDims/dim?
conv1d_transpose/ExpandDims
ExpandDimsinputs(conv1d_transpose/ExpandDims/dim:output:0*
T0*9
_output_shapes'
%:#???????????????????2
conv1d_transpose/ExpandDims?
,conv1d_transpose/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_transpose_expanddims_1_readvariableop_resource*#
_output_shapes
:@?*
dtype02.
,conv1d_transpose/ExpandDims_1/ReadVariableOp?
!conv1d_transpose/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!conv1d_transpose/ExpandDims_1/dim?
conv1d_transpose/ExpandDims_1
ExpandDims4conv1d_transpose/ExpandDims_1/ReadVariableOp:value:0*conv1d_transpose/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:@?2
conv1d_transpose/ExpandDims_1?
$conv1d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$conv1d_transpose/strided_slice/stack?
&conv1d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv1d_transpose/strided_slice/stack_1?
&conv1d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv1d_transpose/strided_slice/stack_2?
conv1d_transpose/strided_sliceStridedSlicestack:output:0-conv1d_transpose/strided_slice/stack:output:0/conv1d_transpose/strided_slice/stack_1:output:0/conv1d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2 
conv1d_transpose/strided_slice?
&conv1d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2(
&conv1d_transpose/strided_slice_1/stack?
(conv1d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(conv1d_transpose/strided_slice_1/stack_1?
(conv1d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv1d_transpose/strided_slice_1/stack_2?
 conv1d_transpose/strided_slice_1StridedSlicestack:output:0/conv1d_transpose/strided_slice_1/stack:output:01conv1d_transpose/strided_slice_1/stack_1:output:01conv1d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2"
 conv1d_transpose/strided_slice_1?
 conv1d_transpose/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 conv1d_transpose/concat/values_1~
conv1d_transpose/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d_transpose/concat/axis?
conv1d_transpose/concatConcatV2'conv1d_transpose/strided_slice:output:0)conv1d_transpose/concat/values_1:output:0)conv1d_transpose/strided_slice_1:output:0%conv1d_transpose/concat/axis:output:0*
N*
T0*
_output_shapes
:2
conv1d_transpose/concat?
conv1d_transposeConv2DBackpropInput conv1d_transpose/concat:output:0&conv1d_transpose/ExpandDims_1:output:0$conv1d_transpose/ExpandDims:output:0*
T0*8
_output_shapes&
$:"??????????????????@*
paddingSAME*
strides
2
conv1d_transpose?
conv1d_transpose/SqueezeSqueezeconv1d_transpose:output:0*
T0*4
_output_shapes"
 :??????????????????@*
squeeze_dims
2
conv1d_transpose/Squeeze?
IdentityIdentity!conv1d_transpose/Squeeze:output:0^NoOp*
T0*4
_output_shapes"
 :??????????????????@2

Identity}
NoOpNoOp-^conv1d_transpose/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:???????????????????: 2\
,conv1d_transpose/ExpandDims_1/ReadVariableOp,conv1d_transpose/ExpandDims_1/ReadVariableOp:] Y
5
_output_shapes#
!:???????????????????
 
_user_specified_nameinputs
?.
?
P__inference_conv1d_transpose_2_layer_call_and_return_conditional_losses_78011402

inputsM
5conv1d_transpose_expanddims_1_readvariableop_resource:??
identity??,conv1d_transpose/ExpandDims_1/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulU
stack/2Const*
_output_shapes
: *
dtype0*
value
B :?2	
stack/2w
stackPackstrided_slice:output:0mul:z:0stack/2:output:0*
N*
T0*
_output_shapes
:2
stack?
conv1d_transpose/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2!
conv1d_transpose/ExpandDims/dim?
conv1d_transpose/ExpandDims
ExpandDimsinputs(conv1d_transpose/ExpandDims/dim:output:0*
T0*1
_output_shapes
:???????????2
conv1d_transpose/ExpandDims?
,conv1d_transpose/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_transpose_expanddims_1_readvariableop_resource*$
_output_shapes
:??*
dtype02.
,conv1d_transpose/ExpandDims_1/ReadVariableOp?
!conv1d_transpose/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!conv1d_transpose/ExpandDims_1/dim?
conv1d_transpose/ExpandDims_1
ExpandDims4conv1d_transpose/ExpandDims_1/ReadVariableOp:value:0*conv1d_transpose/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:??2
conv1d_transpose/ExpandDims_1?
$conv1d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$conv1d_transpose/strided_slice/stack?
&conv1d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv1d_transpose/strided_slice/stack_1?
&conv1d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv1d_transpose/strided_slice/stack_2?
conv1d_transpose/strided_sliceStridedSlicestack:output:0-conv1d_transpose/strided_slice/stack:output:0/conv1d_transpose/strided_slice/stack_1:output:0/conv1d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2 
conv1d_transpose/strided_slice?
&conv1d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2(
&conv1d_transpose/strided_slice_1/stack?
(conv1d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(conv1d_transpose/strided_slice_1/stack_1?
(conv1d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv1d_transpose/strided_slice_1/stack_2?
 conv1d_transpose/strided_slice_1StridedSlicestack:output:0/conv1d_transpose/strided_slice_1/stack:output:01conv1d_transpose/strided_slice_1/stack_1:output:01conv1d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2"
 conv1d_transpose/strided_slice_1?
 conv1d_transpose/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 conv1d_transpose/concat/values_1~
conv1d_transpose/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d_transpose/concat/axis?
conv1d_transpose/concatConcatV2'conv1d_transpose/strided_slice:output:0)conv1d_transpose/concat/values_1:output:0)conv1d_transpose/strided_slice_1:output:0%conv1d_transpose/concat/axis:output:0*
N*
T0*
_output_shapes
:2
conv1d_transpose/concat?
conv1d_transposeConv2DBackpropInput conv1d_transpose/concat:output:0&conv1d_transpose/ExpandDims_1:output:0$conv1d_transpose/ExpandDims:output:0*
T0*9
_output_shapes'
%:#???????????????????*
paddingSAME*
strides
2
conv1d_transpose?
conv1d_transpose/SqueezeSqueezeconv1d_transpose:output:0*
T0*-
_output_shapes
:???????????*
squeeze_dims
2
conv1d_transpose/Squeeze?
IdentityIdentity!conv1d_transpose/Squeeze:output:0^NoOp*
T0*-
_output_shapes
:???????????2

Identity}
NoOpNoOp-^conv1d_transpose/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:???????????: 2\
,conv1d_transpose/ExpandDims_1/ReadVariableOp,conv1d_transpose/ExpandDims_1/ReadVariableOp:U Q
-
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
5__inference_conv1d_transpose_2_layer_call_fn_78012459

inputs
unknown:??
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:???????????????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_conv1d_transpose_2_layer_call_and_return_conditional_losses_780110092
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:???????????????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:???????????????????: 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:???????????????????
 
_user_specified_nameinputs
?
L
0__inference_leaky_re_lu_1_layer_call_fn_78012284

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????@?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_780113192
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:?????????@?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:?????????@?:T P
,
_output_shapes
:?????????@?
 
_user_specified_nameinputs
ҹ
?
J__inference_WG_Generator_layer_call_and_return_conditional_losses_78012112

inputs8
$dense_matmul_readvariableop_resource:
??^
Fconv1d_transpose_conv1d_transpose_expanddims_1_readvariableop_resource:??`
Hconv1d_transpose_1_conv1d_transpose_expanddims_1_readvariableop_resource:??`
Hconv1d_transpose_2_conv1d_transpose_expanddims_1_readvariableop_resource:??_
Hconv1d_transpose_3_conv1d_transpose_expanddims_1_readvariableop_resource:@?^
Hconv1d_transpose_4_conv1d_transpose_expanddims_1_readvariableop_resource:@
identity??=conv1d_transpose/conv1d_transpose/ExpandDims_1/ReadVariableOp??conv1d_transpose_1/conv1d_transpose/ExpandDims_1/ReadVariableOp??conv1d_transpose_2/conv1d_transpose/ExpandDims_1/ReadVariableOp??conv1d_transpose_3/conv1d_transpose/ExpandDims_1/ReadVariableOp??conv1d_transpose_4/conv1d_transpose/ExpandDims_1/ReadVariableOp?dense/MatMul/ReadVariableOp?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
dense/MatMul/ReadVariableOp?
dense/MatMulMatMulinputs#dense/MatMul/ReadVariableOp:value:0*
T0*)
_output_shapes
:???????????2
dense/MatMul?
leaky_re_lu/LeakyRelu	LeakyReludense/MatMul:product:0*)
_output_shapes
:???????????*
alpha%???>2
leaky_re_lu/LeakyReluq
reshape/ShapeShape#leaky_re_lu/LeakyRelu:activations:0*
T0*
_output_shapes
:2
reshape/Shape?
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape/strided_slice/stack?
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_1?
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_2?
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape/strided_slicet
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/1u
reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value
B :?2
reshape/Reshape/shape/2?
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
reshape/Reshape/shape?
reshape/ReshapeReshape#leaky_re_lu/LeakyRelu:activations:0reshape/Reshape/shape:output:0*
T0*,
_output_shapes
:??????????2
reshape/Reshapex
conv1d_transpose/ShapeShapereshape/Reshape:output:0*
T0*
_output_shapes
:2
conv1d_transpose/Shape?
$conv1d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$conv1d_transpose/strided_slice/stack?
&conv1d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv1d_transpose/strided_slice/stack_1?
&conv1d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv1d_transpose/strided_slice/stack_2?
conv1d_transpose/strided_sliceStridedSliceconv1d_transpose/Shape:output:0-conv1d_transpose/strided_slice/stack:output:0/conv1d_transpose/strided_slice/stack_1:output:0/conv1d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
conv1d_transpose/strided_slice?
&conv1d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2(
&conv1d_transpose/strided_slice_1/stack?
(conv1d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv1d_transpose/strided_slice_1/stack_1?
(conv1d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv1d_transpose/strided_slice_1/stack_2?
 conv1d_transpose/strided_slice_1StridedSliceconv1d_transpose/Shape:output:0/conv1d_transpose/strided_slice_1/stack:output:01conv1d_transpose/strided_slice_1/stack_1:output:01conv1d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv1d_transpose/strided_slice_1r
conv1d_transpose/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
conv1d_transpose/mul/y?
conv1d_transpose/mulMul)conv1d_transpose/strided_slice_1:output:0conv1d_transpose/mul/y:output:0*
T0*
_output_shapes
: 2
conv1d_transpose/mulw
conv1d_transpose/stack/2Const*
_output_shapes
: *
dtype0*
value
B :?2
conv1d_transpose/stack/2?
conv1d_transpose/stackPack'conv1d_transpose/strided_slice:output:0conv1d_transpose/mul:z:0!conv1d_transpose/stack/2:output:0*
N*
T0*
_output_shapes
:2
conv1d_transpose/stack?
0conv1d_transpose/conv1d_transpose/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :22
0conv1d_transpose/conv1d_transpose/ExpandDims/dim?
,conv1d_transpose/conv1d_transpose/ExpandDims
ExpandDimsreshape/Reshape:output:09conv1d_transpose/conv1d_transpose/ExpandDims/dim:output:0*
T0*0
_output_shapes
:??????????2.
,conv1d_transpose/conv1d_transpose/ExpandDims?
=conv1d_transpose/conv1d_transpose/ExpandDims_1/ReadVariableOpReadVariableOpFconv1d_transpose_conv1d_transpose_expanddims_1_readvariableop_resource*$
_output_shapes
:??*
dtype02?
=conv1d_transpose/conv1d_transpose/ExpandDims_1/ReadVariableOp?
2conv1d_transpose/conv1d_transpose/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 24
2conv1d_transpose/conv1d_transpose/ExpandDims_1/dim?
.conv1d_transpose/conv1d_transpose/ExpandDims_1
ExpandDimsEconv1d_transpose/conv1d_transpose/ExpandDims_1/ReadVariableOp:value:0;conv1d_transpose/conv1d_transpose/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:??20
.conv1d_transpose/conv1d_transpose/ExpandDims_1?
5conv1d_transpose/conv1d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 27
5conv1d_transpose/conv1d_transpose/strided_slice/stack?
7conv1d_transpose/conv1d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7conv1d_transpose/conv1d_transpose/strided_slice/stack_1?
7conv1d_transpose/conv1d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7conv1d_transpose/conv1d_transpose/strided_slice/stack_2?
/conv1d_transpose/conv1d_transpose/strided_sliceStridedSliceconv1d_transpose/stack:output:0>conv1d_transpose/conv1d_transpose/strided_slice/stack:output:0@conv1d_transpose/conv1d_transpose/strided_slice/stack_1:output:0@conv1d_transpose/conv1d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask21
/conv1d_transpose/conv1d_transpose/strided_slice?
7conv1d_transpose/conv1d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:29
7conv1d_transpose/conv1d_transpose/strided_slice_1/stack?
9conv1d_transpose/conv1d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2;
9conv1d_transpose/conv1d_transpose/strided_slice_1/stack_1?
9conv1d_transpose/conv1d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2;
9conv1d_transpose/conv1d_transpose/strided_slice_1/stack_2?
1conv1d_transpose/conv1d_transpose/strided_slice_1StridedSliceconv1d_transpose/stack:output:0@conv1d_transpose/conv1d_transpose/strided_slice_1/stack:output:0Bconv1d_transpose/conv1d_transpose/strided_slice_1/stack_1:output:0Bconv1d_transpose/conv1d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask23
1conv1d_transpose/conv1d_transpose/strided_slice_1?
1conv1d_transpose/conv1d_transpose/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:23
1conv1d_transpose/conv1d_transpose/concat/values_1?
-conv1d_transpose/conv1d_transpose/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-conv1d_transpose/conv1d_transpose/concat/axis?
(conv1d_transpose/conv1d_transpose/concatConcatV28conv1d_transpose/conv1d_transpose/strided_slice:output:0:conv1d_transpose/conv1d_transpose/concat/values_1:output:0:conv1d_transpose/conv1d_transpose/strided_slice_1:output:06conv1d_transpose/conv1d_transpose/concat/axis:output:0*
N*
T0*
_output_shapes
:2*
(conv1d_transpose/conv1d_transpose/concat?
!conv1d_transpose/conv1d_transposeConv2DBackpropInput1conv1d_transpose/conv1d_transpose/concat:output:07conv1d_transpose/conv1d_transpose/ExpandDims_1:output:05conv1d_transpose/conv1d_transpose/ExpandDims:output:0*
T0*9
_output_shapes'
%:#???????????????????*
paddingSAME*
strides
2#
!conv1d_transpose/conv1d_transpose?
)conv1d_transpose/conv1d_transpose/SqueezeSqueeze*conv1d_transpose/conv1d_transpose:output:0*
T0*,
_output_shapes
:?????????@?*
squeeze_dims
2+
)conv1d_transpose/conv1d_transpose/Squeeze?
leaky_re_lu_1/LeakyRelu	LeakyRelu2conv1d_transpose/conv1d_transpose/Squeeze:output:0*,
_output_shapes
:?????????@?*
alpha%???>2
leaky_re_lu_1/LeakyRelu?
conv1d_transpose_1/ShapeShape%leaky_re_lu_1/LeakyRelu:activations:0*
T0*
_output_shapes
:2
conv1d_transpose_1/Shape?
&conv1d_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv1d_transpose_1/strided_slice/stack?
(conv1d_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv1d_transpose_1/strided_slice/stack_1?
(conv1d_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv1d_transpose_1/strided_slice/stack_2?
 conv1d_transpose_1/strided_sliceStridedSlice!conv1d_transpose_1/Shape:output:0/conv1d_transpose_1/strided_slice/stack:output:01conv1d_transpose_1/strided_slice/stack_1:output:01conv1d_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv1d_transpose_1/strided_slice?
(conv1d_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2*
(conv1d_transpose_1/strided_slice_1/stack?
*conv1d_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv1d_transpose_1/strided_slice_1/stack_1?
*conv1d_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv1d_transpose_1/strided_slice_1/stack_2?
"conv1d_transpose_1/strided_slice_1StridedSlice!conv1d_transpose_1/Shape:output:01conv1d_transpose_1/strided_slice_1/stack:output:03conv1d_transpose_1/strided_slice_1/stack_1:output:03conv1d_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv1d_transpose_1/strided_slice_1v
conv1d_transpose_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
conv1d_transpose_1/mul/y?
conv1d_transpose_1/mulMul+conv1d_transpose_1/strided_slice_1:output:0!conv1d_transpose_1/mul/y:output:0*
T0*
_output_shapes
: 2
conv1d_transpose_1/mul{
conv1d_transpose_1/stack/2Const*
_output_shapes
: *
dtype0*
value
B :?2
conv1d_transpose_1/stack/2?
conv1d_transpose_1/stackPack)conv1d_transpose_1/strided_slice:output:0conv1d_transpose_1/mul:z:0#conv1d_transpose_1/stack/2:output:0*
N*
T0*
_output_shapes
:2
conv1d_transpose_1/stack?
2conv1d_transpose_1/conv1d_transpose/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :24
2conv1d_transpose_1/conv1d_transpose/ExpandDims/dim?
.conv1d_transpose_1/conv1d_transpose/ExpandDims
ExpandDims%leaky_re_lu_1/LeakyRelu:activations:0;conv1d_transpose_1/conv1d_transpose/ExpandDims/dim:output:0*
T0*0
_output_shapes
:?????????@?20
.conv1d_transpose_1/conv1d_transpose/ExpandDims?
?conv1d_transpose_1/conv1d_transpose/ExpandDims_1/ReadVariableOpReadVariableOpHconv1d_transpose_1_conv1d_transpose_expanddims_1_readvariableop_resource*$
_output_shapes
:??*
dtype02A
?conv1d_transpose_1/conv1d_transpose/ExpandDims_1/ReadVariableOp?
4conv1d_transpose_1/conv1d_transpose/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 26
4conv1d_transpose_1/conv1d_transpose/ExpandDims_1/dim?
0conv1d_transpose_1/conv1d_transpose/ExpandDims_1
ExpandDimsGconv1d_transpose_1/conv1d_transpose/ExpandDims_1/ReadVariableOp:value:0=conv1d_transpose_1/conv1d_transpose/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:??22
0conv1d_transpose_1/conv1d_transpose/ExpandDims_1?
7conv1d_transpose_1/conv1d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 29
7conv1d_transpose_1/conv1d_transpose/strided_slice/stack?
9conv1d_transpose_1/conv1d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2;
9conv1d_transpose_1/conv1d_transpose/strided_slice/stack_1?
9conv1d_transpose_1/conv1d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2;
9conv1d_transpose_1/conv1d_transpose/strided_slice/stack_2?
1conv1d_transpose_1/conv1d_transpose/strided_sliceStridedSlice!conv1d_transpose_1/stack:output:0@conv1d_transpose_1/conv1d_transpose/strided_slice/stack:output:0Bconv1d_transpose_1/conv1d_transpose/strided_slice/stack_1:output:0Bconv1d_transpose_1/conv1d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask23
1conv1d_transpose_1/conv1d_transpose/strided_slice?
9conv1d_transpose_1/conv1d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2;
9conv1d_transpose_1/conv1d_transpose/strided_slice_1/stack?
;conv1d_transpose_1/conv1d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2=
;conv1d_transpose_1/conv1d_transpose/strided_slice_1/stack_1?
;conv1d_transpose_1/conv1d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2=
;conv1d_transpose_1/conv1d_transpose/strided_slice_1/stack_2?
3conv1d_transpose_1/conv1d_transpose/strided_slice_1StridedSlice!conv1d_transpose_1/stack:output:0Bconv1d_transpose_1/conv1d_transpose/strided_slice_1/stack:output:0Dconv1d_transpose_1/conv1d_transpose/strided_slice_1/stack_1:output:0Dconv1d_transpose_1/conv1d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask25
3conv1d_transpose_1/conv1d_transpose/strided_slice_1?
3conv1d_transpose_1/conv1d_transpose/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:25
3conv1d_transpose_1/conv1d_transpose/concat/values_1?
/conv1d_transpose_1/conv1d_transpose/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/conv1d_transpose_1/conv1d_transpose/concat/axis?
*conv1d_transpose_1/conv1d_transpose/concatConcatV2:conv1d_transpose_1/conv1d_transpose/strided_slice:output:0<conv1d_transpose_1/conv1d_transpose/concat/values_1:output:0<conv1d_transpose_1/conv1d_transpose/strided_slice_1:output:08conv1d_transpose_1/conv1d_transpose/concat/axis:output:0*
N*
T0*
_output_shapes
:2,
*conv1d_transpose_1/conv1d_transpose/concat?
#conv1d_transpose_1/conv1d_transposeConv2DBackpropInput3conv1d_transpose_1/conv1d_transpose/concat:output:09conv1d_transpose_1/conv1d_transpose/ExpandDims_1:output:07conv1d_transpose_1/conv1d_transpose/ExpandDims:output:0*
T0*9
_output_shapes'
%:#???????????????????*
paddingSAME*
strides
2%
#conv1d_transpose_1/conv1d_transpose?
+conv1d_transpose_1/conv1d_transpose/SqueezeSqueeze,conv1d_transpose_1/conv1d_transpose:output:0*
T0*-
_output_shapes
:???????????*
squeeze_dims
2-
+conv1d_transpose_1/conv1d_transpose/Squeeze?
leaky_re_lu_2/LeakyRelu	LeakyRelu4conv1d_transpose_1/conv1d_transpose/Squeeze:output:0*-
_output_shapes
:???????????*
alpha%???>2
leaky_re_lu_2/LeakyRelu?
conv1d_transpose_2/ShapeShape%leaky_re_lu_2/LeakyRelu:activations:0*
T0*
_output_shapes
:2
conv1d_transpose_2/Shape?
&conv1d_transpose_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv1d_transpose_2/strided_slice/stack?
(conv1d_transpose_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv1d_transpose_2/strided_slice/stack_1?
(conv1d_transpose_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv1d_transpose_2/strided_slice/stack_2?
 conv1d_transpose_2/strided_sliceStridedSlice!conv1d_transpose_2/Shape:output:0/conv1d_transpose_2/strided_slice/stack:output:01conv1d_transpose_2/strided_slice/stack_1:output:01conv1d_transpose_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv1d_transpose_2/strided_slice?
(conv1d_transpose_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2*
(conv1d_transpose_2/strided_slice_1/stack?
*conv1d_transpose_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv1d_transpose_2/strided_slice_1/stack_1?
*conv1d_transpose_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv1d_transpose_2/strided_slice_1/stack_2?
"conv1d_transpose_2/strided_slice_1StridedSlice!conv1d_transpose_2/Shape:output:01conv1d_transpose_2/strided_slice_1/stack:output:03conv1d_transpose_2/strided_slice_1/stack_1:output:03conv1d_transpose_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv1d_transpose_2/strided_slice_1v
conv1d_transpose_2/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
conv1d_transpose_2/mul/y?
conv1d_transpose_2/mulMul+conv1d_transpose_2/strided_slice_1:output:0!conv1d_transpose_2/mul/y:output:0*
T0*
_output_shapes
: 2
conv1d_transpose_2/mul{
conv1d_transpose_2/stack/2Const*
_output_shapes
: *
dtype0*
value
B :?2
conv1d_transpose_2/stack/2?
conv1d_transpose_2/stackPack)conv1d_transpose_2/strided_slice:output:0conv1d_transpose_2/mul:z:0#conv1d_transpose_2/stack/2:output:0*
N*
T0*
_output_shapes
:2
conv1d_transpose_2/stack?
2conv1d_transpose_2/conv1d_transpose/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :24
2conv1d_transpose_2/conv1d_transpose/ExpandDims/dim?
.conv1d_transpose_2/conv1d_transpose/ExpandDims
ExpandDims%leaky_re_lu_2/LeakyRelu:activations:0;conv1d_transpose_2/conv1d_transpose/ExpandDims/dim:output:0*
T0*1
_output_shapes
:???????????20
.conv1d_transpose_2/conv1d_transpose/ExpandDims?
?conv1d_transpose_2/conv1d_transpose/ExpandDims_1/ReadVariableOpReadVariableOpHconv1d_transpose_2_conv1d_transpose_expanddims_1_readvariableop_resource*$
_output_shapes
:??*
dtype02A
?conv1d_transpose_2/conv1d_transpose/ExpandDims_1/ReadVariableOp?
4conv1d_transpose_2/conv1d_transpose/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 26
4conv1d_transpose_2/conv1d_transpose/ExpandDims_1/dim?
0conv1d_transpose_2/conv1d_transpose/ExpandDims_1
ExpandDimsGconv1d_transpose_2/conv1d_transpose/ExpandDims_1/ReadVariableOp:value:0=conv1d_transpose_2/conv1d_transpose/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:??22
0conv1d_transpose_2/conv1d_transpose/ExpandDims_1?
7conv1d_transpose_2/conv1d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 29
7conv1d_transpose_2/conv1d_transpose/strided_slice/stack?
9conv1d_transpose_2/conv1d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2;
9conv1d_transpose_2/conv1d_transpose/strided_slice/stack_1?
9conv1d_transpose_2/conv1d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2;
9conv1d_transpose_2/conv1d_transpose/strided_slice/stack_2?
1conv1d_transpose_2/conv1d_transpose/strided_sliceStridedSlice!conv1d_transpose_2/stack:output:0@conv1d_transpose_2/conv1d_transpose/strided_slice/stack:output:0Bconv1d_transpose_2/conv1d_transpose/strided_slice/stack_1:output:0Bconv1d_transpose_2/conv1d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask23
1conv1d_transpose_2/conv1d_transpose/strided_slice?
9conv1d_transpose_2/conv1d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2;
9conv1d_transpose_2/conv1d_transpose/strided_slice_1/stack?
;conv1d_transpose_2/conv1d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2=
;conv1d_transpose_2/conv1d_transpose/strided_slice_1/stack_1?
;conv1d_transpose_2/conv1d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2=
;conv1d_transpose_2/conv1d_transpose/strided_slice_1/stack_2?
3conv1d_transpose_2/conv1d_transpose/strided_slice_1StridedSlice!conv1d_transpose_2/stack:output:0Bconv1d_transpose_2/conv1d_transpose/strided_slice_1/stack:output:0Dconv1d_transpose_2/conv1d_transpose/strided_slice_1/stack_1:output:0Dconv1d_transpose_2/conv1d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask25
3conv1d_transpose_2/conv1d_transpose/strided_slice_1?
3conv1d_transpose_2/conv1d_transpose/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:25
3conv1d_transpose_2/conv1d_transpose/concat/values_1?
/conv1d_transpose_2/conv1d_transpose/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/conv1d_transpose_2/conv1d_transpose/concat/axis?
*conv1d_transpose_2/conv1d_transpose/concatConcatV2:conv1d_transpose_2/conv1d_transpose/strided_slice:output:0<conv1d_transpose_2/conv1d_transpose/concat/values_1:output:0<conv1d_transpose_2/conv1d_transpose/strided_slice_1:output:08conv1d_transpose_2/conv1d_transpose/concat/axis:output:0*
N*
T0*
_output_shapes
:2,
*conv1d_transpose_2/conv1d_transpose/concat?
#conv1d_transpose_2/conv1d_transposeConv2DBackpropInput3conv1d_transpose_2/conv1d_transpose/concat:output:09conv1d_transpose_2/conv1d_transpose/ExpandDims_1:output:07conv1d_transpose_2/conv1d_transpose/ExpandDims:output:0*
T0*9
_output_shapes'
%:#???????????????????*
paddingSAME*
strides
2%
#conv1d_transpose_2/conv1d_transpose?
+conv1d_transpose_2/conv1d_transpose/SqueezeSqueeze,conv1d_transpose_2/conv1d_transpose:output:0*
T0*-
_output_shapes
:???????????*
squeeze_dims
2-
+conv1d_transpose_2/conv1d_transpose/Squeeze?
leaky_re_lu_3/LeakyRelu	LeakyRelu4conv1d_transpose_2/conv1d_transpose/Squeeze:output:0*-
_output_shapes
:???????????*
alpha%???>2
leaky_re_lu_3/LeakyRelu?
conv1d_transpose_3/ShapeShape%leaky_re_lu_3/LeakyRelu:activations:0*
T0*
_output_shapes
:2
conv1d_transpose_3/Shape?
&conv1d_transpose_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv1d_transpose_3/strided_slice/stack?
(conv1d_transpose_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv1d_transpose_3/strided_slice/stack_1?
(conv1d_transpose_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv1d_transpose_3/strided_slice/stack_2?
 conv1d_transpose_3/strided_sliceStridedSlice!conv1d_transpose_3/Shape:output:0/conv1d_transpose_3/strided_slice/stack:output:01conv1d_transpose_3/strided_slice/stack_1:output:01conv1d_transpose_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv1d_transpose_3/strided_slice?
(conv1d_transpose_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2*
(conv1d_transpose_3/strided_slice_1/stack?
*conv1d_transpose_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv1d_transpose_3/strided_slice_1/stack_1?
*conv1d_transpose_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv1d_transpose_3/strided_slice_1/stack_2?
"conv1d_transpose_3/strided_slice_1StridedSlice!conv1d_transpose_3/Shape:output:01conv1d_transpose_3/strided_slice_1/stack:output:03conv1d_transpose_3/strided_slice_1/stack_1:output:03conv1d_transpose_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv1d_transpose_3/strided_slice_1v
conv1d_transpose_3/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
conv1d_transpose_3/mul/y?
conv1d_transpose_3/mulMul+conv1d_transpose_3/strided_slice_1:output:0!conv1d_transpose_3/mul/y:output:0*
T0*
_output_shapes
: 2
conv1d_transpose_3/mulz
conv1d_transpose_3/stack/2Const*
_output_shapes
: *
dtype0*
value	B :@2
conv1d_transpose_3/stack/2?
conv1d_transpose_3/stackPack)conv1d_transpose_3/strided_slice:output:0conv1d_transpose_3/mul:z:0#conv1d_transpose_3/stack/2:output:0*
N*
T0*
_output_shapes
:2
conv1d_transpose_3/stack?
2conv1d_transpose_3/conv1d_transpose/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :24
2conv1d_transpose_3/conv1d_transpose/ExpandDims/dim?
.conv1d_transpose_3/conv1d_transpose/ExpandDims
ExpandDims%leaky_re_lu_3/LeakyRelu:activations:0;conv1d_transpose_3/conv1d_transpose/ExpandDims/dim:output:0*
T0*1
_output_shapes
:???????????20
.conv1d_transpose_3/conv1d_transpose/ExpandDims?
?conv1d_transpose_3/conv1d_transpose/ExpandDims_1/ReadVariableOpReadVariableOpHconv1d_transpose_3_conv1d_transpose_expanddims_1_readvariableop_resource*#
_output_shapes
:@?*
dtype02A
?conv1d_transpose_3/conv1d_transpose/ExpandDims_1/ReadVariableOp?
4conv1d_transpose_3/conv1d_transpose/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 26
4conv1d_transpose_3/conv1d_transpose/ExpandDims_1/dim?
0conv1d_transpose_3/conv1d_transpose/ExpandDims_1
ExpandDimsGconv1d_transpose_3/conv1d_transpose/ExpandDims_1/ReadVariableOp:value:0=conv1d_transpose_3/conv1d_transpose/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:@?22
0conv1d_transpose_3/conv1d_transpose/ExpandDims_1?
7conv1d_transpose_3/conv1d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 29
7conv1d_transpose_3/conv1d_transpose/strided_slice/stack?
9conv1d_transpose_3/conv1d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2;
9conv1d_transpose_3/conv1d_transpose/strided_slice/stack_1?
9conv1d_transpose_3/conv1d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2;
9conv1d_transpose_3/conv1d_transpose/strided_slice/stack_2?
1conv1d_transpose_3/conv1d_transpose/strided_sliceStridedSlice!conv1d_transpose_3/stack:output:0@conv1d_transpose_3/conv1d_transpose/strided_slice/stack:output:0Bconv1d_transpose_3/conv1d_transpose/strided_slice/stack_1:output:0Bconv1d_transpose_3/conv1d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask23
1conv1d_transpose_3/conv1d_transpose/strided_slice?
9conv1d_transpose_3/conv1d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2;
9conv1d_transpose_3/conv1d_transpose/strided_slice_1/stack?
;conv1d_transpose_3/conv1d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2=
;conv1d_transpose_3/conv1d_transpose/strided_slice_1/stack_1?
;conv1d_transpose_3/conv1d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2=
;conv1d_transpose_3/conv1d_transpose/strided_slice_1/stack_2?
3conv1d_transpose_3/conv1d_transpose/strided_slice_1StridedSlice!conv1d_transpose_3/stack:output:0Bconv1d_transpose_3/conv1d_transpose/strided_slice_1/stack:output:0Dconv1d_transpose_3/conv1d_transpose/strided_slice_1/stack_1:output:0Dconv1d_transpose_3/conv1d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask25
3conv1d_transpose_3/conv1d_transpose/strided_slice_1?
3conv1d_transpose_3/conv1d_transpose/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:25
3conv1d_transpose_3/conv1d_transpose/concat/values_1?
/conv1d_transpose_3/conv1d_transpose/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/conv1d_transpose_3/conv1d_transpose/concat/axis?
*conv1d_transpose_3/conv1d_transpose/concatConcatV2:conv1d_transpose_3/conv1d_transpose/strided_slice:output:0<conv1d_transpose_3/conv1d_transpose/concat/values_1:output:0<conv1d_transpose_3/conv1d_transpose/strided_slice_1:output:08conv1d_transpose_3/conv1d_transpose/concat/axis:output:0*
N*
T0*
_output_shapes
:2,
*conv1d_transpose_3/conv1d_transpose/concat?
#conv1d_transpose_3/conv1d_transposeConv2DBackpropInput3conv1d_transpose_3/conv1d_transpose/concat:output:09conv1d_transpose_3/conv1d_transpose/ExpandDims_1:output:07conv1d_transpose_3/conv1d_transpose/ExpandDims:output:0*
T0*8
_output_shapes&
$:"??????????????????@*
paddingSAME*
strides
2%
#conv1d_transpose_3/conv1d_transpose?
+conv1d_transpose_3/conv1d_transpose/SqueezeSqueeze,conv1d_transpose_3/conv1d_transpose:output:0*
T0*,
_output_shapes
:?????????? @*
squeeze_dims
2-
+conv1d_transpose_3/conv1d_transpose/Squeeze?
leaky_re_lu_4/LeakyRelu	LeakyRelu4conv1d_transpose_3/conv1d_transpose/Squeeze:output:0*,
_output_shapes
:?????????? @*
alpha%???>2
leaky_re_lu_4/LeakyRelu?
conv1d_transpose_4/ShapeShape%leaky_re_lu_4/LeakyRelu:activations:0*
T0*
_output_shapes
:2
conv1d_transpose_4/Shape?
&conv1d_transpose_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv1d_transpose_4/strided_slice/stack?
(conv1d_transpose_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv1d_transpose_4/strided_slice/stack_1?
(conv1d_transpose_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv1d_transpose_4/strided_slice/stack_2?
 conv1d_transpose_4/strided_sliceStridedSlice!conv1d_transpose_4/Shape:output:0/conv1d_transpose_4/strided_slice/stack:output:01conv1d_transpose_4/strided_slice/stack_1:output:01conv1d_transpose_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv1d_transpose_4/strided_slice?
(conv1d_transpose_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2*
(conv1d_transpose_4/strided_slice_1/stack?
*conv1d_transpose_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv1d_transpose_4/strided_slice_1/stack_1?
*conv1d_transpose_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv1d_transpose_4/strided_slice_1/stack_2?
"conv1d_transpose_4/strided_slice_1StridedSlice!conv1d_transpose_4/Shape:output:01conv1d_transpose_4/strided_slice_1/stack:output:03conv1d_transpose_4/strided_slice_1/stack_1:output:03conv1d_transpose_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv1d_transpose_4/strided_slice_1v
conv1d_transpose_4/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
conv1d_transpose_4/mul/y?
conv1d_transpose_4/mulMul+conv1d_transpose_4/strided_slice_1:output:0!conv1d_transpose_4/mul/y:output:0*
T0*
_output_shapes
: 2
conv1d_transpose_4/mulz
conv1d_transpose_4/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
conv1d_transpose_4/stack/2?
conv1d_transpose_4/stackPack)conv1d_transpose_4/strided_slice:output:0conv1d_transpose_4/mul:z:0#conv1d_transpose_4/stack/2:output:0*
N*
T0*
_output_shapes
:2
conv1d_transpose_4/stack?
2conv1d_transpose_4/conv1d_transpose/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :24
2conv1d_transpose_4/conv1d_transpose/ExpandDims/dim?
.conv1d_transpose_4/conv1d_transpose/ExpandDims
ExpandDims%leaky_re_lu_4/LeakyRelu:activations:0;conv1d_transpose_4/conv1d_transpose/ExpandDims/dim:output:0*
T0*0
_output_shapes
:?????????? @20
.conv1d_transpose_4/conv1d_transpose/ExpandDims?
?conv1d_transpose_4/conv1d_transpose/ExpandDims_1/ReadVariableOpReadVariableOpHconv1d_transpose_4_conv1d_transpose_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02A
?conv1d_transpose_4/conv1d_transpose/ExpandDims_1/ReadVariableOp?
4conv1d_transpose_4/conv1d_transpose/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 26
4conv1d_transpose_4/conv1d_transpose/ExpandDims_1/dim?
0conv1d_transpose_4/conv1d_transpose/ExpandDims_1
ExpandDimsGconv1d_transpose_4/conv1d_transpose/ExpandDims_1/ReadVariableOp:value:0=conv1d_transpose_4/conv1d_transpose/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@22
0conv1d_transpose_4/conv1d_transpose/ExpandDims_1?
7conv1d_transpose_4/conv1d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 29
7conv1d_transpose_4/conv1d_transpose/strided_slice/stack?
9conv1d_transpose_4/conv1d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2;
9conv1d_transpose_4/conv1d_transpose/strided_slice/stack_1?
9conv1d_transpose_4/conv1d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2;
9conv1d_transpose_4/conv1d_transpose/strided_slice/stack_2?
1conv1d_transpose_4/conv1d_transpose/strided_sliceStridedSlice!conv1d_transpose_4/stack:output:0@conv1d_transpose_4/conv1d_transpose/strided_slice/stack:output:0Bconv1d_transpose_4/conv1d_transpose/strided_slice/stack_1:output:0Bconv1d_transpose_4/conv1d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask23
1conv1d_transpose_4/conv1d_transpose/strided_slice?
9conv1d_transpose_4/conv1d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2;
9conv1d_transpose_4/conv1d_transpose/strided_slice_1/stack?
;conv1d_transpose_4/conv1d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2=
;conv1d_transpose_4/conv1d_transpose/strided_slice_1/stack_1?
;conv1d_transpose_4/conv1d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2=
;conv1d_transpose_4/conv1d_transpose/strided_slice_1/stack_2?
3conv1d_transpose_4/conv1d_transpose/strided_slice_1StridedSlice!conv1d_transpose_4/stack:output:0Bconv1d_transpose_4/conv1d_transpose/strided_slice_1/stack:output:0Dconv1d_transpose_4/conv1d_transpose/strided_slice_1/stack_1:output:0Dconv1d_transpose_4/conv1d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask25
3conv1d_transpose_4/conv1d_transpose/strided_slice_1?
3conv1d_transpose_4/conv1d_transpose/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:25
3conv1d_transpose_4/conv1d_transpose/concat/values_1?
/conv1d_transpose_4/conv1d_transpose/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/conv1d_transpose_4/conv1d_transpose/concat/axis?
*conv1d_transpose_4/conv1d_transpose/concatConcatV2:conv1d_transpose_4/conv1d_transpose/strided_slice:output:0<conv1d_transpose_4/conv1d_transpose/concat/values_1:output:0<conv1d_transpose_4/conv1d_transpose/strided_slice_1:output:08conv1d_transpose_4/conv1d_transpose/concat/axis:output:0*
N*
T0*
_output_shapes
:2,
*conv1d_transpose_4/conv1d_transpose/concat?
#conv1d_transpose_4/conv1d_transposeConv2DBackpropInput3conv1d_transpose_4/conv1d_transpose/concat:output:09conv1d_transpose_4/conv1d_transpose/ExpandDims_1:output:07conv1d_transpose_4/conv1d_transpose/ExpandDims:output:0*
T0*8
_output_shapes&
$:"??????????????????*
paddingSAME*
strides
2%
#conv1d_transpose_4/conv1d_transpose?
+conv1d_transpose_4/conv1d_transpose/SqueezeSqueeze,conv1d_transpose_4/conv1d_transpose:output:0*
T0*-
_output_shapes
:???????????*
squeeze_dims
2-
+conv1d_transpose_4/conv1d_transpose/Squeeze?
conv1d_transpose_4/TanhTanh4conv1d_transpose_4/conv1d_transpose/Squeeze:output:0*
T0*-
_output_shapes
:???????????2
conv1d_transpose_4/Tanh|
IdentityIdentityconv1d_transpose_4/Tanh:y:0^NoOp*
T0*-
_output_shapes
:???????????2

Identity?
NoOpNoOp>^conv1d_transpose/conv1d_transpose/ExpandDims_1/ReadVariableOp@^conv1d_transpose_1/conv1d_transpose/ExpandDims_1/ReadVariableOp@^conv1d_transpose_2/conv1d_transpose/ExpandDims_1/ReadVariableOp@^conv1d_transpose_3/conv1d_transpose/ExpandDims_1/ReadVariableOp@^conv1d_transpose_4/conv1d_transpose/ExpandDims_1/ReadVariableOp^dense/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : : : : : 2~
=conv1d_transpose/conv1d_transpose/ExpandDims_1/ReadVariableOp=conv1d_transpose/conv1d_transpose/ExpandDims_1/ReadVariableOp2?
?conv1d_transpose_1/conv1d_transpose/ExpandDims_1/ReadVariableOp?conv1d_transpose_1/conv1d_transpose/ExpandDims_1/ReadVariableOp2?
?conv1d_transpose_2/conv1d_transpose/ExpandDims_1/ReadVariableOp?conv1d_transpose_2/conv1d_transpose/ExpandDims_1/ReadVariableOp2?
?conv1d_transpose_3/conv1d_transpose/ExpandDims_1/ReadVariableOp?conv1d_transpose_3/conv1d_transpose/ExpandDims_1/ReadVariableOp2?
?conv1d_transpose_4/conv1d_transpose/ExpandDims_1/ReadVariableOp?conv1d_transpose_4/conv1d_transpose/ExpandDims_1/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
g
K__inference_leaky_re_lu_4_layer_call_and_return_conditional_losses_78012567

inputs
identityi
	LeakyRelu	LeakyReluinputs*,
_output_shapes
:?????????? @*
alpha%???>2
	LeakyRelup
IdentityIdentityLeakyRelu:activations:0*
T0*,
_output_shapes
:?????????? @2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:?????????? @:T P
,
_output_shapes
:?????????? @
 
_user_specified_nameinputs
?.
?
P__inference_conv1d_transpose_3_layer_call_and_return_conditional_losses_78012512

inputsL
5conv1d_transpose_expanddims_1_readvariableop_resource:@?
identity??,conv1d_transpose/ExpandDims_1/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulT
stack/2Const*
_output_shapes
: *
dtype0*
value	B :@2	
stack/2w
stackPackstrided_slice:output:0mul:z:0stack/2:output:0*
N*
T0*
_output_shapes
:2
stack?
conv1d_transpose/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2!
conv1d_transpose/ExpandDims/dim?
conv1d_transpose/ExpandDims
ExpandDimsinputs(conv1d_transpose/ExpandDims/dim:output:0*
T0*9
_output_shapes'
%:#???????????????????2
conv1d_transpose/ExpandDims?
,conv1d_transpose/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_transpose_expanddims_1_readvariableop_resource*#
_output_shapes
:@?*
dtype02.
,conv1d_transpose/ExpandDims_1/ReadVariableOp?
!conv1d_transpose/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!conv1d_transpose/ExpandDims_1/dim?
conv1d_transpose/ExpandDims_1
ExpandDims4conv1d_transpose/ExpandDims_1/ReadVariableOp:value:0*conv1d_transpose/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:@?2
conv1d_transpose/ExpandDims_1?
$conv1d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$conv1d_transpose/strided_slice/stack?
&conv1d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv1d_transpose/strided_slice/stack_1?
&conv1d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv1d_transpose/strided_slice/stack_2?
conv1d_transpose/strided_sliceStridedSlicestack:output:0-conv1d_transpose/strided_slice/stack:output:0/conv1d_transpose/strided_slice/stack_1:output:0/conv1d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2 
conv1d_transpose/strided_slice?
&conv1d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2(
&conv1d_transpose/strided_slice_1/stack?
(conv1d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(conv1d_transpose/strided_slice_1/stack_1?
(conv1d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv1d_transpose/strided_slice_1/stack_2?
 conv1d_transpose/strided_slice_1StridedSlicestack:output:0/conv1d_transpose/strided_slice_1/stack:output:01conv1d_transpose/strided_slice_1/stack_1:output:01conv1d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2"
 conv1d_transpose/strided_slice_1?
 conv1d_transpose/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 conv1d_transpose/concat/values_1~
conv1d_transpose/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d_transpose/concat/axis?
conv1d_transpose/concatConcatV2'conv1d_transpose/strided_slice:output:0)conv1d_transpose/concat/values_1:output:0)conv1d_transpose/strided_slice_1:output:0%conv1d_transpose/concat/axis:output:0*
N*
T0*
_output_shapes
:2
conv1d_transpose/concat?
conv1d_transposeConv2DBackpropInput conv1d_transpose/concat:output:0&conv1d_transpose/ExpandDims_1:output:0$conv1d_transpose/ExpandDims:output:0*
T0*8
_output_shapes&
$:"??????????????????@*
paddingSAME*
strides
2
conv1d_transpose?
conv1d_transpose/SqueezeSqueezeconv1d_transpose:output:0*
T0*4
_output_shapes"
 :??????????????????@*
squeeze_dims
2
conv1d_transpose/Squeeze?
IdentityIdentity!conv1d_transpose/Squeeze:output:0^NoOp*
T0*4
_output_shapes"
 :??????????????????@2

Identity}
NoOpNoOp-^conv1d_transpose/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:???????????????????: 2\
,conv1d_transpose/ExpandDims_1/ReadVariableOp,conv1d_transpose/ExpandDims_1/ReadVariableOp:] Y
5
_output_shapes#
!:???????????????????
 
_user_specified_nameinputs
?
a
E__inference_reshape_layer_call_and_return_conditional_losses_78011273

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliced
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/1e
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value
B :?2
Reshape/shape/2?
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapet
ReshapeReshapeinputsReshape/shape:output:0*
T0*,
_output_shapes
:??????????2	
Reshapei
IdentityIdentityReshape:output:0*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:???????????:Q M
)
_output_shapes
:???????????
 
_user_specified_nameinputs
?
g
K__inference_leaky_re_lu_3_layer_call_and_return_conditional_losses_78012471

inputs
identityj
	LeakyRelu	LeakyReluinputs*-
_output_shapes
:???????????*
alpha%???>2
	LeakyReluq
IdentityIdentityLeakyRelu:activations:0*
T0*-
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:???????????:U Q
-
_output_shapes
:???????????
 
_user_specified_nameinputs
?
e
I__inference_leaky_re_lu_layer_call_and_return_conditional_losses_78011258

inputs
identityf
	LeakyRelu	LeakyReluinputs*)
_output_shapes
:???????????*
alpha%???>2
	LeakyRelum
IdentityIdentityLeakyRelu:activations:0*
T0*)
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:???????????:Q M
)
_output_shapes
:???????????
 
_user_specified_nameinputs
?
g
K__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_78011319

inputs
identityi
	LeakyRelu	LeakyReluinputs*,
_output_shapes
:?????????@?*
alpha%???>2
	LeakyRelup
IdentityIdentityLeakyRelu:activations:0*
T0*,
_output_shapes
:?????????@?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:?????????@?:T P
,
_output_shapes
:?????????@?
 
_user_specified_nameinputs
?
?
5__inference_conv1d_transpose_4_layer_call_fn_78012653

inputs
unknown:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_conv1d_transpose_4_layer_call_and_return_conditional_losses_780111862
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :??????????????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":??????????????????@: 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :??????????????????@
 
_user_specified_nameinputs
?
?
3__inference_conv1d_transpose_layer_call_fn_78012274

inputs
unknown:??
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????@?*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_conv1d_transpose_layer_call_and_return_conditional_losses_780113102
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:?????????@?2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*-
_input_shapes
:??????????: 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
? 
?
$__inference__traced_restore_78012729
file_prefix1
assignvariableop_dense_kernel:
??B
*assignvariableop_1_conv1d_transpose_kernel:??D
,assignvariableop_2_conv1d_transpose_1_kernel:??D
,assignvariableop_3_conv1d_transpose_2_kernel:??C
,assignvariableop_4_conv1d_transpose_3_kernel:@?B
,assignvariableop_5_conv1d_transpose_4_kernel:@

identity_7??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_2?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*!
valueBB B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*0
_output_shapes
:::::::*
dtypes
	22
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOpassignvariableop_dense_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp*assignvariableop_1_conv1d_transpose_kernelIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp,assignvariableop_2_conv1d_transpose_1_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp,assignvariableop_3_conv1d_transpose_2_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp,assignvariableop_4_conv1d_transpose_3_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp,assignvariableop_5_conv1d_transpose_4_kernelIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?

Identity_6Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_6c

Identity_7IdentityIdentity_6:output:0^NoOp_1*
T0*
_output_shapes
: 2

Identity_7?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5*"
_acd_function_control_output(*
_output_shapes
 2
NoOp_1"!

identity_7Identity_7:output:0*!
_input_shapes
: : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_5:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
g
K__inference_leaky_re_lu_4_layer_call_and_return_conditional_losses_78011457

inputs
identityi
	LeakyRelu	LeakyReluinputs*,
_output_shapes
:?????????? @*
alpha%???>2
	LeakyRelup
IdentityIdentityLeakyRelu:activations:0*
T0*,
_output_shapes
:?????????? @2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:?????????? @:T P
,
_output_shapes
:?????????? @
 
_user_specified_nameinputs
?/
?
P__inference_conv1d_transpose_4_layer_call_and_return_conditional_losses_78012646

inputsK
5conv1d_transpose_expanddims_1_readvariableop_resource:@
identity??,conv1d_transpose/ExpandDims_1/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulT
stack/2Const*
_output_shapes
: *
dtype0*
value	B :2	
stack/2w
stackPackstrided_slice:output:0mul:z:0stack/2:output:0*
N*
T0*
_output_shapes
:2
stack?
conv1d_transpose/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2!
conv1d_transpose/ExpandDims/dim?
conv1d_transpose/ExpandDims
ExpandDimsinputs(conv1d_transpose/ExpandDims/dim:output:0*
T0*0
_output_shapes
:?????????? @2
conv1d_transpose/ExpandDims?
,conv1d_transpose/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_transpose_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02.
,conv1d_transpose/ExpandDims_1/ReadVariableOp?
!conv1d_transpose/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!conv1d_transpose/ExpandDims_1/dim?
conv1d_transpose/ExpandDims_1
ExpandDims4conv1d_transpose/ExpandDims_1/ReadVariableOp:value:0*conv1d_transpose/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@2
conv1d_transpose/ExpandDims_1?
$conv1d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$conv1d_transpose/strided_slice/stack?
&conv1d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv1d_transpose/strided_slice/stack_1?
&conv1d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv1d_transpose/strided_slice/stack_2?
conv1d_transpose/strided_sliceStridedSlicestack:output:0-conv1d_transpose/strided_slice/stack:output:0/conv1d_transpose/strided_slice/stack_1:output:0/conv1d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2 
conv1d_transpose/strided_slice?
&conv1d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2(
&conv1d_transpose/strided_slice_1/stack?
(conv1d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(conv1d_transpose/strided_slice_1/stack_1?
(conv1d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv1d_transpose/strided_slice_1/stack_2?
 conv1d_transpose/strided_slice_1StridedSlicestack:output:0/conv1d_transpose/strided_slice_1/stack:output:01conv1d_transpose/strided_slice_1/stack_1:output:01conv1d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2"
 conv1d_transpose/strided_slice_1?
 conv1d_transpose/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 conv1d_transpose/concat/values_1~
conv1d_transpose/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d_transpose/concat/axis?
conv1d_transpose/concatConcatV2'conv1d_transpose/strided_slice:output:0)conv1d_transpose/concat/values_1:output:0)conv1d_transpose/strided_slice_1:output:0%conv1d_transpose/concat/axis:output:0*
N*
T0*
_output_shapes
:2
conv1d_transpose/concat?
conv1d_transposeConv2DBackpropInput conv1d_transpose/concat:output:0&conv1d_transpose/ExpandDims_1:output:0$conv1d_transpose/ExpandDims:output:0*
T0*8
_output_shapes&
$:"??????????????????*
paddingSAME*
strides
2
conv1d_transpose?
conv1d_transpose/SqueezeSqueezeconv1d_transpose:output:0*
T0*-
_output_shapes
:???????????*
squeeze_dims
2
conv1d_transpose/Squeezeo
TanhTanh!conv1d_transpose/Squeeze:output:0*
T0*-
_output_shapes
:???????????2
Tanhi
IdentityIdentityTanh:y:0^NoOp*
T0*-
_output_shapes
:???????????2

Identity}
NoOpNoOp-^conv1d_transpose/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*-
_input_shapes
:?????????? @: 2\
,conv1d_transpose/ExpandDims_1/ReadVariableOp,conv1d_transpose/ExpandDims_1/ReadVariableOp:T P
,
_output_shapes
:?????????? @
 
_user_specified_nameinputs
?.
?
P__inference_conv1d_transpose_1_layer_call_and_return_conditional_losses_78012320

inputsM
5conv1d_transpose_expanddims_1_readvariableop_resource:??
identity??,conv1d_transpose/ExpandDims_1/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulU
stack/2Const*
_output_shapes
: *
dtype0*
value
B :?2	
stack/2w
stackPackstrided_slice:output:0mul:z:0stack/2:output:0*
N*
T0*
_output_shapes
:2
stack?
conv1d_transpose/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2!
conv1d_transpose/ExpandDims/dim?
conv1d_transpose/ExpandDims
ExpandDimsinputs(conv1d_transpose/ExpandDims/dim:output:0*
T0*9
_output_shapes'
%:#???????????????????2
conv1d_transpose/ExpandDims?
,conv1d_transpose/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_transpose_expanddims_1_readvariableop_resource*$
_output_shapes
:??*
dtype02.
,conv1d_transpose/ExpandDims_1/ReadVariableOp?
!conv1d_transpose/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!conv1d_transpose/ExpandDims_1/dim?
conv1d_transpose/ExpandDims_1
ExpandDims4conv1d_transpose/ExpandDims_1/ReadVariableOp:value:0*conv1d_transpose/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:??2
conv1d_transpose/ExpandDims_1?
$conv1d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$conv1d_transpose/strided_slice/stack?
&conv1d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv1d_transpose/strided_slice/stack_1?
&conv1d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv1d_transpose/strided_slice/stack_2?
conv1d_transpose/strided_sliceStridedSlicestack:output:0-conv1d_transpose/strided_slice/stack:output:0/conv1d_transpose/strided_slice/stack_1:output:0/conv1d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2 
conv1d_transpose/strided_slice?
&conv1d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2(
&conv1d_transpose/strided_slice_1/stack?
(conv1d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(conv1d_transpose/strided_slice_1/stack_1?
(conv1d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv1d_transpose/strided_slice_1/stack_2?
 conv1d_transpose/strided_slice_1StridedSlicestack:output:0/conv1d_transpose/strided_slice_1/stack:output:01conv1d_transpose/strided_slice_1/stack_1:output:01conv1d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2"
 conv1d_transpose/strided_slice_1?
 conv1d_transpose/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 conv1d_transpose/concat/values_1~
conv1d_transpose/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d_transpose/concat/axis?
conv1d_transpose/concatConcatV2'conv1d_transpose/strided_slice:output:0)conv1d_transpose/concat/values_1:output:0)conv1d_transpose/strided_slice_1:output:0%conv1d_transpose/concat/axis:output:0*
N*
T0*
_output_shapes
:2
conv1d_transpose/concat?
conv1d_transposeConv2DBackpropInput conv1d_transpose/concat:output:0&conv1d_transpose/ExpandDims_1:output:0$conv1d_transpose/ExpandDims:output:0*
T0*9
_output_shapes'
%:#???????????????????*
paddingSAME*
strides
2
conv1d_transpose?
conv1d_transpose/SqueezeSqueezeconv1d_transpose:output:0*
T0*5
_output_shapes#
!:???????????????????*
squeeze_dims
2
conv1d_transpose/Squeeze?
IdentityIdentity!conv1d_transpose/Squeeze:output:0^NoOp*
T0*5
_output_shapes#
!:???????????????????2

Identity}
NoOpNoOp-^conv1d_transpose/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:???????????????????: 2\
,conv1d_transpose/ExpandDims_1/ReadVariableOp,conv1d_transpose/ExpandDims_1/ReadVariableOp:] Y
5
_output_shapes#
!:???????????????????
 
_user_specified_nameinputs
?.
?
N__inference_conv1d_transpose_layer_call_and_return_conditional_losses_78012224

inputsM
5conv1d_transpose_expanddims_1_readvariableop_resource:??
identity??,conv1d_transpose/ExpandDims_1/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulU
stack/2Const*
_output_shapes
: *
dtype0*
value
B :?2	
stack/2w
stackPackstrided_slice:output:0mul:z:0stack/2:output:0*
N*
T0*
_output_shapes
:2
stack?
conv1d_transpose/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2!
conv1d_transpose/ExpandDims/dim?
conv1d_transpose/ExpandDims
ExpandDimsinputs(conv1d_transpose/ExpandDims/dim:output:0*
T0*9
_output_shapes'
%:#???????????????????2
conv1d_transpose/ExpandDims?
,conv1d_transpose/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_transpose_expanddims_1_readvariableop_resource*$
_output_shapes
:??*
dtype02.
,conv1d_transpose/ExpandDims_1/ReadVariableOp?
!conv1d_transpose/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!conv1d_transpose/ExpandDims_1/dim?
conv1d_transpose/ExpandDims_1
ExpandDims4conv1d_transpose/ExpandDims_1/ReadVariableOp:value:0*conv1d_transpose/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:??2
conv1d_transpose/ExpandDims_1?
$conv1d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$conv1d_transpose/strided_slice/stack?
&conv1d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv1d_transpose/strided_slice/stack_1?
&conv1d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv1d_transpose/strided_slice/stack_2?
conv1d_transpose/strided_sliceStridedSlicestack:output:0-conv1d_transpose/strided_slice/stack:output:0/conv1d_transpose/strided_slice/stack_1:output:0/conv1d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2 
conv1d_transpose/strided_slice?
&conv1d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2(
&conv1d_transpose/strided_slice_1/stack?
(conv1d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(conv1d_transpose/strided_slice_1/stack_1?
(conv1d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv1d_transpose/strided_slice_1/stack_2?
 conv1d_transpose/strided_slice_1StridedSlicestack:output:0/conv1d_transpose/strided_slice_1/stack:output:01conv1d_transpose/strided_slice_1/stack_1:output:01conv1d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2"
 conv1d_transpose/strided_slice_1?
 conv1d_transpose/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 conv1d_transpose/concat/values_1~
conv1d_transpose/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d_transpose/concat/axis?
conv1d_transpose/concatConcatV2'conv1d_transpose/strided_slice:output:0)conv1d_transpose/concat/values_1:output:0)conv1d_transpose/strided_slice_1:output:0%conv1d_transpose/concat/axis:output:0*
N*
T0*
_output_shapes
:2
conv1d_transpose/concat?
conv1d_transposeConv2DBackpropInput conv1d_transpose/concat:output:0&conv1d_transpose/ExpandDims_1:output:0$conv1d_transpose/ExpandDims:output:0*
T0*9
_output_shapes'
%:#???????????????????*
paddingSAME*
strides
2
conv1d_transpose?
conv1d_transpose/SqueezeSqueezeconv1d_transpose:output:0*
T0*5
_output_shapes#
!:???????????????????*
squeeze_dims
2
conv1d_transpose/Squeeze?
IdentityIdentity!conv1d_transpose/Squeeze:output:0^NoOp*
T0*5
_output_shapes#
!:???????????????????2

Identity}
NoOpNoOp-^conv1d_transpose/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:???????????????????: 2\
,conv1d_transpose/ExpandDims_1/ReadVariableOp,conv1d_transpose/ExpandDims_1/ReadVariableOp:] Y
5
_output_shapes#
!:???????????????????
 
_user_specified_nameinputs
?	
?
/__inference_WG_Generator_layer_call_fn_78012146

inputs
unknown:
??!
	unknown_0:??!
	unknown_1:??!
	unknown_2:?? 
	unknown_3:@?
	unknown_4:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:???????????*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_WG_Generator_layer_call_and_return_conditional_losses_780116412
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*-
_output_shapes
:???????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
a
E__inference_reshape_layer_call_and_return_conditional_losses_78012183

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliced
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/1e
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value
B :?2
Reshape/shape/2?
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapet
ReshapeReshapeinputsReshape/shape:output:0*
T0*,
_output_shapes
:??????????2	
Reshapei
IdentityIdentityReshape:output:0*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:???????????:Q M
)
_output_shapes
:???????????
 
_user_specified_nameinputs
?.
?
N__inference_conv1d_transpose_layer_call_and_return_conditional_losses_78012260

inputsM
5conv1d_transpose_expanddims_1_readvariableop_resource:??
identity??,conv1d_transpose/ExpandDims_1/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulU
stack/2Const*
_output_shapes
: *
dtype0*
value
B :?2	
stack/2w
stackPackstrided_slice:output:0mul:z:0stack/2:output:0*
N*
T0*
_output_shapes
:2
stack?
conv1d_transpose/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2!
conv1d_transpose/ExpandDims/dim?
conv1d_transpose/ExpandDims
ExpandDimsinputs(conv1d_transpose/ExpandDims/dim:output:0*
T0*0
_output_shapes
:??????????2
conv1d_transpose/ExpandDims?
,conv1d_transpose/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_transpose_expanddims_1_readvariableop_resource*$
_output_shapes
:??*
dtype02.
,conv1d_transpose/ExpandDims_1/ReadVariableOp?
!conv1d_transpose/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!conv1d_transpose/ExpandDims_1/dim?
conv1d_transpose/ExpandDims_1
ExpandDims4conv1d_transpose/ExpandDims_1/ReadVariableOp:value:0*conv1d_transpose/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:??2
conv1d_transpose/ExpandDims_1?
$conv1d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$conv1d_transpose/strided_slice/stack?
&conv1d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv1d_transpose/strided_slice/stack_1?
&conv1d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv1d_transpose/strided_slice/stack_2?
conv1d_transpose/strided_sliceStridedSlicestack:output:0-conv1d_transpose/strided_slice/stack:output:0/conv1d_transpose/strided_slice/stack_1:output:0/conv1d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2 
conv1d_transpose/strided_slice?
&conv1d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2(
&conv1d_transpose/strided_slice_1/stack?
(conv1d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(conv1d_transpose/strided_slice_1/stack_1?
(conv1d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv1d_transpose/strided_slice_1/stack_2?
 conv1d_transpose/strided_slice_1StridedSlicestack:output:0/conv1d_transpose/strided_slice_1/stack:output:01conv1d_transpose/strided_slice_1/stack_1:output:01conv1d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2"
 conv1d_transpose/strided_slice_1?
 conv1d_transpose/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 conv1d_transpose/concat/values_1~
conv1d_transpose/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d_transpose/concat/axis?
conv1d_transpose/concatConcatV2'conv1d_transpose/strided_slice:output:0)conv1d_transpose/concat/values_1:output:0)conv1d_transpose/strided_slice_1:output:0%conv1d_transpose/concat/axis:output:0*
N*
T0*
_output_shapes
:2
conv1d_transpose/concat?
conv1d_transposeConv2DBackpropInput conv1d_transpose/concat:output:0&conv1d_transpose/ExpandDims_1:output:0$conv1d_transpose/ExpandDims:output:0*
T0*9
_output_shapes'
%:#???????????????????*
paddingSAME*
strides
2
conv1d_transpose?
conv1d_transpose/SqueezeSqueezeconv1d_transpose:output:0*
T0*,
_output_shapes
:?????????@?*
squeeze_dims
2
conv1d_transpose/Squeeze?
IdentityIdentity!conv1d_transpose/Squeeze:output:0^NoOp*
T0*,
_output_shapes
:?????????@?2

Identity}
NoOpNoOp-^conv1d_transpose/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*-
_input_shapes
:??????????: 2\
,conv1d_transpose/ExpandDims_1/ReadVariableOp,conv1d_transpose/ExpandDims_1/ReadVariableOp:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
L
0__inference_leaky_re_lu_2_layer_call_fn_78012380

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_780113652
PartitionedCallr
IdentityIdentityPartitionedCall:output:0*
T0*-
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:???????????:U Q
-
_output_shapes
:???????????
 
_user_specified_nameinputs
?
g
K__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_78012375

inputs
identityj
	LeakyRelu	LeakyReluinputs*-
_output_shapes
:???????????*
alpha%???>2
	LeakyReluq
IdentityIdentityLeakyRelu:activations:0*
T0*-
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:???????????:U Q
-
_output_shapes
:???????????
 
_user_specified_nameinputs
?.
?
P__inference_conv1d_transpose_1_layer_call_and_return_conditional_losses_78010921

inputsM
5conv1d_transpose_expanddims_1_readvariableop_resource:??
identity??,conv1d_transpose/ExpandDims_1/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulU
stack/2Const*
_output_shapes
: *
dtype0*
value
B :?2	
stack/2w
stackPackstrided_slice:output:0mul:z:0stack/2:output:0*
N*
T0*
_output_shapes
:2
stack?
conv1d_transpose/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2!
conv1d_transpose/ExpandDims/dim?
conv1d_transpose/ExpandDims
ExpandDimsinputs(conv1d_transpose/ExpandDims/dim:output:0*
T0*9
_output_shapes'
%:#???????????????????2
conv1d_transpose/ExpandDims?
,conv1d_transpose/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_transpose_expanddims_1_readvariableop_resource*$
_output_shapes
:??*
dtype02.
,conv1d_transpose/ExpandDims_1/ReadVariableOp?
!conv1d_transpose/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!conv1d_transpose/ExpandDims_1/dim?
conv1d_transpose/ExpandDims_1
ExpandDims4conv1d_transpose/ExpandDims_1/ReadVariableOp:value:0*conv1d_transpose/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:??2
conv1d_transpose/ExpandDims_1?
$conv1d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$conv1d_transpose/strided_slice/stack?
&conv1d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv1d_transpose/strided_slice/stack_1?
&conv1d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv1d_transpose/strided_slice/stack_2?
conv1d_transpose/strided_sliceStridedSlicestack:output:0-conv1d_transpose/strided_slice/stack:output:0/conv1d_transpose/strided_slice/stack_1:output:0/conv1d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2 
conv1d_transpose/strided_slice?
&conv1d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2(
&conv1d_transpose/strided_slice_1/stack?
(conv1d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(conv1d_transpose/strided_slice_1/stack_1?
(conv1d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv1d_transpose/strided_slice_1/stack_2?
 conv1d_transpose/strided_slice_1StridedSlicestack:output:0/conv1d_transpose/strided_slice_1/stack:output:01conv1d_transpose/strided_slice_1/stack_1:output:01conv1d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2"
 conv1d_transpose/strided_slice_1?
 conv1d_transpose/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 conv1d_transpose/concat/values_1~
conv1d_transpose/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d_transpose/concat/axis?
conv1d_transpose/concatConcatV2'conv1d_transpose/strided_slice:output:0)conv1d_transpose/concat/values_1:output:0)conv1d_transpose/strided_slice_1:output:0%conv1d_transpose/concat/axis:output:0*
N*
T0*
_output_shapes
:2
conv1d_transpose/concat?
conv1d_transposeConv2DBackpropInput conv1d_transpose/concat:output:0&conv1d_transpose/ExpandDims_1:output:0$conv1d_transpose/ExpandDims:output:0*
T0*9
_output_shapes'
%:#???????????????????*
paddingSAME*
strides
2
conv1d_transpose?
conv1d_transpose/SqueezeSqueezeconv1d_transpose:output:0*
T0*5
_output_shapes#
!:???????????????????*
squeeze_dims
2
conv1d_transpose/Squeeze?
IdentityIdentity!conv1d_transpose/Squeeze:output:0^NoOp*
T0*5
_output_shapes#
!:???????????????????2

Identity}
NoOpNoOp-^conv1d_transpose/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:???????????????????: 2\
,conv1d_transpose/ExpandDims_1/ReadVariableOp,conv1d_transpose/ExpandDims_1/ReadVariableOp:] Y
5
_output_shapes#
!:???????????????????
 
_user_specified_nameinputs
?.
?
P__inference_conv1d_transpose_2_layer_call_and_return_conditional_losses_78012452

inputsM
5conv1d_transpose_expanddims_1_readvariableop_resource:??
identity??,conv1d_transpose/ExpandDims_1/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulU
stack/2Const*
_output_shapes
: *
dtype0*
value
B :?2	
stack/2w
stackPackstrided_slice:output:0mul:z:0stack/2:output:0*
N*
T0*
_output_shapes
:2
stack?
conv1d_transpose/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2!
conv1d_transpose/ExpandDims/dim?
conv1d_transpose/ExpandDims
ExpandDimsinputs(conv1d_transpose/ExpandDims/dim:output:0*
T0*1
_output_shapes
:???????????2
conv1d_transpose/ExpandDims?
,conv1d_transpose/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_transpose_expanddims_1_readvariableop_resource*$
_output_shapes
:??*
dtype02.
,conv1d_transpose/ExpandDims_1/ReadVariableOp?
!conv1d_transpose/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!conv1d_transpose/ExpandDims_1/dim?
conv1d_transpose/ExpandDims_1
ExpandDims4conv1d_transpose/ExpandDims_1/ReadVariableOp:value:0*conv1d_transpose/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:??2
conv1d_transpose/ExpandDims_1?
$conv1d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$conv1d_transpose/strided_slice/stack?
&conv1d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv1d_transpose/strided_slice/stack_1?
&conv1d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv1d_transpose/strided_slice/stack_2?
conv1d_transpose/strided_sliceStridedSlicestack:output:0-conv1d_transpose/strided_slice/stack:output:0/conv1d_transpose/strided_slice/stack_1:output:0/conv1d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2 
conv1d_transpose/strided_slice?
&conv1d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2(
&conv1d_transpose/strided_slice_1/stack?
(conv1d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(conv1d_transpose/strided_slice_1/stack_1?
(conv1d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv1d_transpose/strided_slice_1/stack_2?
 conv1d_transpose/strided_slice_1StridedSlicestack:output:0/conv1d_transpose/strided_slice_1/stack:output:01conv1d_transpose/strided_slice_1/stack_1:output:01conv1d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2"
 conv1d_transpose/strided_slice_1?
 conv1d_transpose/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 conv1d_transpose/concat/values_1~
conv1d_transpose/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d_transpose/concat/axis?
conv1d_transpose/concatConcatV2'conv1d_transpose/strided_slice:output:0)conv1d_transpose/concat/values_1:output:0)conv1d_transpose/strided_slice_1:output:0%conv1d_transpose/concat/axis:output:0*
N*
T0*
_output_shapes
:2
conv1d_transpose/concat?
conv1d_transposeConv2DBackpropInput conv1d_transpose/concat:output:0&conv1d_transpose/ExpandDims_1:output:0$conv1d_transpose/ExpandDims:output:0*
T0*9
_output_shapes'
%:#???????????????????*
paddingSAME*
strides
2
conv1d_transpose?
conv1d_transpose/SqueezeSqueezeconv1d_transpose:output:0*
T0*-
_output_shapes
:???????????*
squeeze_dims
2
conv1d_transpose/Squeeze?
IdentityIdentity!conv1d_transpose/Squeeze:output:0^NoOp*
T0*-
_output_shapes
:???????????2

Identity}
NoOpNoOp-^conv1d_transpose/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:???????????: 2\
,conv1d_transpose/ExpandDims_1/ReadVariableOp,conv1d_transpose/ExpandDims_1/ReadVariableOp:U Q
-
_output_shapes
:???????????
 
_user_specified_nameinputs
?.
?
P__inference_conv1d_transpose_2_layer_call_and_return_conditional_losses_78011009

inputsM
5conv1d_transpose_expanddims_1_readvariableop_resource:??
identity??,conv1d_transpose/ExpandDims_1/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulU
stack/2Const*
_output_shapes
: *
dtype0*
value
B :?2	
stack/2w
stackPackstrided_slice:output:0mul:z:0stack/2:output:0*
N*
T0*
_output_shapes
:2
stack?
conv1d_transpose/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2!
conv1d_transpose/ExpandDims/dim?
conv1d_transpose/ExpandDims
ExpandDimsinputs(conv1d_transpose/ExpandDims/dim:output:0*
T0*9
_output_shapes'
%:#???????????????????2
conv1d_transpose/ExpandDims?
,conv1d_transpose/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_transpose_expanddims_1_readvariableop_resource*$
_output_shapes
:??*
dtype02.
,conv1d_transpose/ExpandDims_1/ReadVariableOp?
!conv1d_transpose/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!conv1d_transpose/ExpandDims_1/dim?
conv1d_transpose/ExpandDims_1
ExpandDims4conv1d_transpose/ExpandDims_1/ReadVariableOp:value:0*conv1d_transpose/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:??2
conv1d_transpose/ExpandDims_1?
$conv1d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$conv1d_transpose/strided_slice/stack?
&conv1d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv1d_transpose/strided_slice/stack_1?
&conv1d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv1d_transpose/strided_slice/stack_2?
conv1d_transpose/strided_sliceStridedSlicestack:output:0-conv1d_transpose/strided_slice/stack:output:0/conv1d_transpose/strided_slice/stack_1:output:0/conv1d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2 
conv1d_transpose/strided_slice?
&conv1d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2(
&conv1d_transpose/strided_slice_1/stack?
(conv1d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(conv1d_transpose/strided_slice_1/stack_1?
(conv1d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv1d_transpose/strided_slice_1/stack_2?
 conv1d_transpose/strided_slice_1StridedSlicestack:output:0/conv1d_transpose/strided_slice_1/stack:output:01conv1d_transpose/strided_slice_1/stack_1:output:01conv1d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2"
 conv1d_transpose/strided_slice_1?
 conv1d_transpose/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 conv1d_transpose/concat/values_1~
conv1d_transpose/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d_transpose/concat/axis?
conv1d_transpose/concatConcatV2'conv1d_transpose/strided_slice:output:0)conv1d_transpose/concat/values_1:output:0)conv1d_transpose/strided_slice_1:output:0%conv1d_transpose/concat/axis:output:0*
N*
T0*
_output_shapes
:2
conv1d_transpose/concat?
conv1d_transposeConv2DBackpropInput conv1d_transpose/concat:output:0&conv1d_transpose/ExpandDims_1:output:0$conv1d_transpose/ExpandDims:output:0*
T0*9
_output_shapes'
%:#???????????????????*
paddingSAME*
strides
2
conv1d_transpose?
conv1d_transpose/SqueezeSqueezeconv1d_transpose:output:0*
T0*5
_output_shapes#
!:???????????????????*
squeeze_dims
2
conv1d_transpose/Squeeze?
IdentityIdentity!conv1d_transpose/Squeeze:output:0^NoOp*
T0*5
_output_shapes#
!:???????????????????2

Identity}
NoOpNoOp-^conv1d_transpose/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:???????????????????: 2\
,conv1d_transpose/ExpandDims_1/ReadVariableOp,conv1d_transpose/ExpandDims_1/ReadVariableOp:] Y
5
_output_shapes#
!:???????????????????
 
_user_specified_nameinputs
?
?
5__inference_conv1d_transpose_3_layer_call_fn_78012555

inputs
unknown:@?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????@*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_conv1d_transpose_3_layer_call_and_return_conditional_losses_780110972
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :??????????????????@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:???????????????????: 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:???????????????????
 
_user_specified_nameinputs
?
F
*__inference_reshape_layer_call_fn_78012188

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_reshape_layer_call_and_return_conditional_losses_780112732
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:???????????:Q M
)
_output_shapes
:???????????
 
_user_specified_nameinputs
?6
?
J__inference_WG_Generator_layer_call_and_return_conditional_losses_78011729
input_1"
dense_78011704:
??1
conv1d_transpose_78011709:??3
conv1d_transpose_1_78011713:??3
conv1d_transpose_2_78011717:??2
conv1d_transpose_3_78011721:@?1
conv1d_transpose_4_78011725:@
identity??(conv1d_transpose/StatefulPartitionedCall?*conv1d_transpose_1/StatefulPartitionedCall?*conv1d_transpose_2/StatefulPartitionedCall?*conv1d_transpose_3/StatefulPartitionedCall?*conv1d_transpose_4/StatefulPartitionedCall?dense/StatefulPartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCallinput_1dense_78011704*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_layer_call_and_return_conditional_losses_780112492
dense/StatefulPartitionedCall?
leaky_re_lu/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_leaky_re_lu_layer_call_and_return_conditional_losses_780112582
leaky_re_lu/PartitionedCall?
reshape/PartitionedCallPartitionedCall$leaky_re_lu/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_reshape_layer_call_and_return_conditional_losses_780112732
reshape/PartitionedCall?
(conv1d_transpose/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0conv1d_transpose_78011709*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????@?*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_conv1d_transpose_layer_call_and_return_conditional_losses_780113102*
(conv1d_transpose/StatefulPartitionedCall?
leaky_re_lu_1/PartitionedCallPartitionedCall1conv1d_transpose/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????@?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_780113192
leaky_re_lu_1/PartitionedCall?
*conv1d_transpose_1/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_1/PartitionedCall:output:0conv1d_transpose_1_78011713*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:???????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_conv1d_transpose_1_layer_call_and_return_conditional_losses_780113562,
*conv1d_transpose_1/StatefulPartitionedCall?
leaky_re_lu_2/PartitionedCallPartitionedCall3conv1d_transpose_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_780113652
leaky_re_lu_2/PartitionedCall?
*conv1d_transpose_2/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_2/PartitionedCall:output:0conv1d_transpose_2_78011717*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:???????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_conv1d_transpose_2_layer_call_and_return_conditional_losses_780114022,
*conv1d_transpose_2/StatefulPartitionedCall?
leaky_re_lu_3/PartitionedCallPartitionedCall3conv1d_transpose_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_leaky_re_lu_3_layer_call_and_return_conditional_losses_780114112
leaky_re_lu_3/PartitionedCall?
*conv1d_transpose_3/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_3/PartitionedCall:output:0conv1d_transpose_3_78011721*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????? @*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_conv1d_transpose_3_layer_call_and_return_conditional_losses_780114482,
*conv1d_transpose_3/StatefulPartitionedCall?
leaky_re_lu_4/PartitionedCallPartitionedCall3conv1d_transpose_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????? @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_leaky_re_lu_4_layer_call_and_return_conditional_losses_780114572
leaky_re_lu_4/PartitionedCall?
*conv1d_transpose_4/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_4/PartitionedCall:output:0conv1d_transpose_4_78011725*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:???????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_conv1d_transpose_4_layer_call_and_return_conditional_losses_780114952,
*conv1d_transpose_4/StatefulPartitionedCall?
IdentityIdentity3conv1d_transpose_4/StatefulPartitionedCall:output:0^NoOp*
T0*-
_output_shapes
:???????????2

Identity?
NoOpNoOp)^conv1d_transpose/StatefulPartitionedCall+^conv1d_transpose_1/StatefulPartitionedCall+^conv1d_transpose_2/StatefulPartitionedCall+^conv1d_transpose_3/StatefulPartitionedCall+^conv1d_transpose_4/StatefulPartitionedCall^dense/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : : : : : 2T
(conv1d_transpose/StatefulPartitionedCall(conv1d_transpose/StatefulPartitionedCall2X
*conv1d_transpose_1/StatefulPartitionedCall*conv1d_transpose_1/StatefulPartitionedCall2X
*conv1d_transpose_2/StatefulPartitionedCall*conv1d_transpose_2/StatefulPartitionedCall2X
*conv1d_transpose_3/StatefulPartitionedCall*conv1d_transpose_3/StatefulPartitionedCall2X
*conv1d_transpose_4/StatefulPartitionedCall*conv1d_transpose_4/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1
?
L
0__inference_leaky_re_lu_4_layer_call_fn_78012572

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????? @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_leaky_re_lu_4_layer_call_and_return_conditional_losses_780114572
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:?????????? @2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:?????????? @:T P
,
_output_shapes
:?????????? @
 
_user_specified_nameinputs
?
e
I__inference_leaky_re_lu_layer_call_and_return_conditional_losses_78012165

inputs
identityf
	LeakyRelu	LeakyReluinputs*)
_output_shapes
:???????????*
alpha%???>2
	LeakyRelum
IdentityIdentityLeakyRelu:activations:0*
T0*)
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:???????????:Q M
)
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
!__inference__traced_save_78012701
file_prefix+
'savev2_dense_kernel_read_readvariableop6
2savev2_conv1d_transpose_kernel_read_readvariableop8
4savev2_conv1d_transpose_1_kernel_read_readvariableop8
4savev2_conv1d_transpose_2_kernel_read_readvariableop8
4savev2_conv1d_transpose_3_kernel_read_readvariableop8
4savev2_conv1d_transpose_4_kernel_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*!
valueBB B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0'savev2_dense_kernel_read_readvariableop2savev2_conv1d_transpose_kernel_read_readvariableop4savev2_conv1d_transpose_1_kernel_read_readvariableop4savev2_conv1d_transpose_2_kernel_read_readvariableop4savev2_conv1d_transpose_3_kernel_read_readvariableop4savev2_conv1d_transpose_4_kernel_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
	22
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: 2

Identity_1c
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"!

identity_1Identity_1:output:0*p
_input_shapes_
]: :
??:??:??:??:@?:@: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:&"
 
_output_shapes
:
??:*&
$
_output_shapes
:??:*&
$
_output_shapes
:??:*&
$
_output_shapes
:??:)%
#
_output_shapes
:@?:($
"
_output_shapes
:@:

_output_shapes
: 
?
?
5__inference_conv1d_transpose_1_layer_call_fn_78012363

inputs
unknown:??
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:???????????????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_conv1d_transpose_1_layer_call_and_return_conditional_losses_780109212
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:???????????????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:???????????????????: 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:???????????????????
 
_user_specified_nameinputs
?.
?
P__inference_conv1d_transpose_3_layer_call_and_return_conditional_losses_78012548

inputsL
5conv1d_transpose_expanddims_1_readvariableop_resource:@?
identity??,conv1d_transpose/ExpandDims_1/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulT
stack/2Const*
_output_shapes
: *
dtype0*
value	B :@2	
stack/2w
stackPackstrided_slice:output:0mul:z:0stack/2:output:0*
N*
T0*
_output_shapes
:2
stack?
conv1d_transpose/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2!
conv1d_transpose/ExpandDims/dim?
conv1d_transpose/ExpandDims
ExpandDimsinputs(conv1d_transpose/ExpandDims/dim:output:0*
T0*1
_output_shapes
:???????????2
conv1d_transpose/ExpandDims?
,conv1d_transpose/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_transpose_expanddims_1_readvariableop_resource*#
_output_shapes
:@?*
dtype02.
,conv1d_transpose/ExpandDims_1/ReadVariableOp?
!conv1d_transpose/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!conv1d_transpose/ExpandDims_1/dim?
conv1d_transpose/ExpandDims_1
ExpandDims4conv1d_transpose/ExpandDims_1/ReadVariableOp:value:0*conv1d_transpose/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:@?2
conv1d_transpose/ExpandDims_1?
$conv1d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$conv1d_transpose/strided_slice/stack?
&conv1d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv1d_transpose/strided_slice/stack_1?
&conv1d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv1d_transpose/strided_slice/stack_2?
conv1d_transpose/strided_sliceStridedSlicestack:output:0-conv1d_transpose/strided_slice/stack:output:0/conv1d_transpose/strided_slice/stack_1:output:0/conv1d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2 
conv1d_transpose/strided_slice?
&conv1d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2(
&conv1d_transpose/strided_slice_1/stack?
(conv1d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(conv1d_transpose/strided_slice_1/stack_1?
(conv1d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv1d_transpose/strided_slice_1/stack_2?
 conv1d_transpose/strided_slice_1StridedSlicestack:output:0/conv1d_transpose/strided_slice_1/stack:output:01conv1d_transpose/strided_slice_1/stack_1:output:01conv1d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2"
 conv1d_transpose/strided_slice_1?
 conv1d_transpose/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 conv1d_transpose/concat/values_1~
conv1d_transpose/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d_transpose/concat/axis?
conv1d_transpose/concatConcatV2'conv1d_transpose/strided_slice:output:0)conv1d_transpose/concat/values_1:output:0)conv1d_transpose/strided_slice_1:output:0%conv1d_transpose/concat/axis:output:0*
N*
T0*
_output_shapes
:2
conv1d_transpose/concat?
conv1d_transposeConv2DBackpropInput conv1d_transpose/concat:output:0&conv1d_transpose/ExpandDims_1:output:0$conv1d_transpose/ExpandDims:output:0*
T0*8
_output_shapes&
$:"??????????????????@*
paddingSAME*
strides
2
conv1d_transpose?
conv1d_transpose/SqueezeSqueezeconv1d_transpose:output:0*
T0*,
_output_shapes
:?????????? @*
squeeze_dims
2
conv1d_transpose/Squeeze?
IdentityIdentity!conv1d_transpose/Squeeze:output:0^NoOp*
T0*,
_output_shapes
:?????????? @2

Identity}
NoOpNoOp-^conv1d_transpose/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:???????????: 2\
,conv1d_transpose/ExpandDims_1/ReadVariableOp,conv1d_transpose/ExpandDims_1/ReadVariableOp:U Q
-
_output_shapes
:???????????
 
_user_specified_nameinputs
??
?
#__inference__wrapped_model_78010793
input_1E
1wg_generator_dense_matmul_readvariableop_resource:
??k
Swg_generator_conv1d_transpose_conv1d_transpose_expanddims_1_readvariableop_resource:??m
Uwg_generator_conv1d_transpose_1_conv1d_transpose_expanddims_1_readvariableop_resource:??m
Uwg_generator_conv1d_transpose_2_conv1d_transpose_expanddims_1_readvariableop_resource:??l
Uwg_generator_conv1d_transpose_3_conv1d_transpose_expanddims_1_readvariableop_resource:@?k
Uwg_generator_conv1d_transpose_4_conv1d_transpose_expanddims_1_readvariableop_resource:@
identity??JWG_Generator/conv1d_transpose/conv1d_transpose/ExpandDims_1/ReadVariableOp?LWG_Generator/conv1d_transpose_1/conv1d_transpose/ExpandDims_1/ReadVariableOp?LWG_Generator/conv1d_transpose_2/conv1d_transpose/ExpandDims_1/ReadVariableOp?LWG_Generator/conv1d_transpose_3/conv1d_transpose/ExpandDims_1/ReadVariableOp?LWG_Generator/conv1d_transpose_4/conv1d_transpose/ExpandDims_1/ReadVariableOp?(WG_Generator/dense/MatMul/ReadVariableOp?
(WG_Generator/dense/MatMul/ReadVariableOpReadVariableOp1wg_generator_dense_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02*
(WG_Generator/dense/MatMul/ReadVariableOp?
WG_Generator/dense/MatMulMatMulinput_10WG_Generator/dense/MatMul/ReadVariableOp:value:0*
T0*)
_output_shapes
:???????????2
WG_Generator/dense/MatMul?
"WG_Generator/leaky_re_lu/LeakyRelu	LeakyRelu#WG_Generator/dense/MatMul:product:0*)
_output_shapes
:???????????*
alpha%???>2$
"WG_Generator/leaky_re_lu/LeakyRelu?
WG_Generator/reshape/ShapeShape0WG_Generator/leaky_re_lu/LeakyRelu:activations:0*
T0*
_output_shapes
:2
WG_Generator/reshape/Shape?
(WG_Generator/reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(WG_Generator/reshape/strided_slice/stack?
*WG_Generator/reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*WG_Generator/reshape/strided_slice/stack_1?
*WG_Generator/reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*WG_Generator/reshape/strided_slice/stack_2?
"WG_Generator/reshape/strided_sliceStridedSlice#WG_Generator/reshape/Shape:output:01WG_Generator/reshape/strided_slice/stack:output:03WG_Generator/reshape/strided_slice/stack_1:output:03WG_Generator/reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"WG_Generator/reshape/strided_slice?
$WG_Generator/reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2&
$WG_Generator/reshape/Reshape/shape/1?
$WG_Generator/reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value
B :?2&
$WG_Generator/reshape/Reshape/shape/2?
"WG_Generator/reshape/Reshape/shapePack+WG_Generator/reshape/strided_slice:output:0-WG_Generator/reshape/Reshape/shape/1:output:0-WG_Generator/reshape/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2$
"WG_Generator/reshape/Reshape/shape?
WG_Generator/reshape/ReshapeReshape0WG_Generator/leaky_re_lu/LeakyRelu:activations:0+WG_Generator/reshape/Reshape/shape:output:0*
T0*,
_output_shapes
:??????????2
WG_Generator/reshape/Reshape?
#WG_Generator/conv1d_transpose/ShapeShape%WG_Generator/reshape/Reshape:output:0*
T0*
_output_shapes
:2%
#WG_Generator/conv1d_transpose/Shape?
1WG_Generator/conv1d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 23
1WG_Generator/conv1d_transpose/strided_slice/stack?
3WG_Generator/conv1d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:25
3WG_Generator/conv1d_transpose/strided_slice/stack_1?
3WG_Generator/conv1d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3WG_Generator/conv1d_transpose/strided_slice/stack_2?
+WG_Generator/conv1d_transpose/strided_sliceStridedSlice,WG_Generator/conv1d_transpose/Shape:output:0:WG_Generator/conv1d_transpose/strided_slice/stack:output:0<WG_Generator/conv1d_transpose/strided_slice/stack_1:output:0<WG_Generator/conv1d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2-
+WG_Generator/conv1d_transpose/strided_slice?
3WG_Generator/conv1d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:25
3WG_Generator/conv1d_transpose/strided_slice_1/stack?
5WG_Generator/conv1d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5WG_Generator/conv1d_transpose/strided_slice_1/stack_1?
5WG_Generator/conv1d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5WG_Generator/conv1d_transpose/strided_slice_1/stack_2?
-WG_Generator/conv1d_transpose/strided_slice_1StridedSlice,WG_Generator/conv1d_transpose/Shape:output:0<WG_Generator/conv1d_transpose/strided_slice_1/stack:output:0>WG_Generator/conv1d_transpose/strided_slice_1/stack_1:output:0>WG_Generator/conv1d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-WG_Generator/conv1d_transpose/strided_slice_1?
#WG_Generator/conv1d_transpose/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2%
#WG_Generator/conv1d_transpose/mul/y?
!WG_Generator/conv1d_transpose/mulMul6WG_Generator/conv1d_transpose/strided_slice_1:output:0,WG_Generator/conv1d_transpose/mul/y:output:0*
T0*
_output_shapes
: 2#
!WG_Generator/conv1d_transpose/mul?
%WG_Generator/conv1d_transpose/stack/2Const*
_output_shapes
: *
dtype0*
value
B :?2'
%WG_Generator/conv1d_transpose/stack/2?
#WG_Generator/conv1d_transpose/stackPack4WG_Generator/conv1d_transpose/strided_slice:output:0%WG_Generator/conv1d_transpose/mul:z:0.WG_Generator/conv1d_transpose/stack/2:output:0*
N*
T0*
_output_shapes
:2%
#WG_Generator/conv1d_transpose/stack?
=WG_Generator/conv1d_transpose/conv1d_transpose/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2?
=WG_Generator/conv1d_transpose/conv1d_transpose/ExpandDims/dim?
9WG_Generator/conv1d_transpose/conv1d_transpose/ExpandDims
ExpandDims%WG_Generator/reshape/Reshape:output:0FWG_Generator/conv1d_transpose/conv1d_transpose/ExpandDims/dim:output:0*
T0*0
_output_shapes
:??????????2;
9WG_Generator/conv1d_transpose/conv1d_transpose/ExpandDims?
JWG_Generator/conv1d_transpose/conv1d_transpose/ExpandDims_1/ReadVariableOpReadVariableOpSwg_generator_conv1d_transpose_conv1d_transpose_expanddims_1_readvariableop_resource*$
_output_shapes
:??*
dtype02L
JWG_Generator/conv1d_transpose/conv1d_transpose/ExpandDims_1/ReadVariableOp?
?WG_Generator/conv1d_transpose/conv1d_transpose/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2A
?WG_Generator/conv1d_transpose/conv1d_transpose/ExpandDims_1/dim?
;WG_Generator/conv1d_transpose/conv1d_transpose/ExpandDims_1
ExpandDimsRWG_Generator/conv1d_transpose/conv1d_transpose/ExpandDims_1/ReadVariableOp:value:0HWG_Generator/conv1d_transpose/conv1d_transpose/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:??2=
;WG_Generator/conv1d_transpose/conv1d_transpose/ExpandDims_1?
BWG_Generator/conv1d_transpose/conv1d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2D
BWG_Generator/conv1d_transpose/conv1d_transpose/strided_slice/stack?
DWG_Generator/conv1d_transpose/conv1d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2F
DWG_Generator/conv1d_transpose/conv1d_transpose/strided_slice/stack_1?
DWG_Generator/conv1d_transpose/conv1d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2F
DWG_Generator/conv1d_transpose/conv1d_transpose/strided_slice/stack_2?
<WG_Generator/conv1d_transpose/conv1d_transpose/strided_sliceStridedSlice,WG_Generator/conv1d_transpose/stack:output:0KWG_Generator/conv1d_transpose/conv1d_transpose/strided_slice/stack:output:0MWG_Generator/conv1d_transpose/conv1d_transpose/strided_slice/stack_1:output:0MWG_Generator/conv1d_transpose/conv1d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2>
<WG_Generator/conv1d_transpose/conv1d_transpose/strided_slice?
DWG_Generator/conv1d_transpose/conv1d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2F
DWG_Generator/conv1d_transpose/conv1d_transpose/strided_slice_1/stack?
FWG_Generator/conv1d_transpose/conv1d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2H
FWG_Generator/conv1d_transpose/conv1d_transpose/strided_slice_1/stack_1?
FWG_Generator/conv1d_transpose/conv1d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2H
FWG_Generator/conv1d_transpose/conv1d_transpose/strided_slice_1/stack_2?
>WG_Generator/conv1d_transpose/conv1d_transpose/strided_slice_1StridedSlice,WG_Generator/conv1d_transpose/stack:output:0MWG_Generator/conv1d_transpose/conv1d_transpose/strided_slice_1/stack:output:0OWG_Generator/conv1d_transpose/conv1d_transpose/strided_slice_1/stack_1:output:0OWG_Generator/conv1d_transpose/conv1d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2@
>WG_Generator/conv1d_transpose/conv1d_transpose/strided_slice_1?
>WG_Generator/conv1d_transpose/conv1d_transpose/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:2@
>WG_Generator/conv1d_transpose/conv1d_transpose/concat/values_1?
:WG_Generator/conv1d_transpose/conv1d_transpose/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2<
:WG_Generator/conv1d_transpose/conv1d_transpose/concat/axis?
5WG_Generator/conv1d_transpose/conv1d_transpose/concatConcatV2EWG_Generator/conv1d_transpose/conv1d_transpose/strided_slice:output:0GWG_Generator/conv1d_transpose/conv1d_transpose/concat/values_1:output:0GWG_Generator/conv1d_transpose/conv1d_transpose/strided_slice_1:output:0CWG_Generator/conv1d_transpose/conv1d_transpose/concat/axis:output:0*
N*
T0*
_output_shapes
:27
5WG_Generator/conv1d_transpose/conv1d_transpose/concat?
.WG_Generator/conv1d_transpose/conv1d_transposeConv2DBackpropInput>WG_Generator/conv1d_transpose/conv1d_transpose/concat:output:0DWG_Generator/conv1d_transpose/conv1d_transpose/ExpandDims_1:output:0BWG_Generator/conv1d_transpose/conv1d_transpose/ExpandDims:output:0*
T0*9
_output_shapes'
%:#???????????????????*
paddingSAME*
strides
20
.WG_Generator/conv1d_transpose/conv1d_transpose?
6WG_Generator/conv1d_transpose/conv1d_transpose/SqueezeSqueeze7WG_Generator/conv1d_transpose/conv1d_transpose:output:0*
T0*,
_output_shapes
:?????????@?*
squeeze_dims
28
6WG_Generator/conv1d_transpose/conv1d_transpose/Squeeze?
$WG_Generator/leaky_re_lu_1/LeakyRelu	LeakyRelu?WG_Generator/conv1d_transpose/conv1d_transpose/Squeeze:output:0*,
_output_shapes
:?????????@?*
alpha%???>2&
$WG_Generator/leaky_re_lu_1/LeakyRelu?
%WG_Generator/conv1d_transpose_1/ShapeShape2WG_Generator/leaky_re_lu_1/LeakyRelu:activations:0*
T0*
_output_shapes
:2'
%WG_Generator/conv1d_transpose_1/Shape?
3WG_Generator/conv1d_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 25
3WG_Generator/conv1d_transpose_1/strided_slice/stack?
5WG_Generator/conv1d_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5WG_Generator/conv1d_transpose_1/strided_slice/stack_1?
5WG_Generator/conv1d_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5WG_Generator/conv1d_transpose_1/strided_slice/stack_2?
-WG_Generator/conv1d_transpose_1/strided_sliceStridedSlice.WG_Generator/conv1d_transpose_1/Shape:output:0<WG_Generator/conv1d_transpose_1/strided_slice/stack:output:0>WG_Generator/conv1d_transpose_1/strided_slice/stack_1:output:0>WG_Generator/conv1d_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-WG_Generator/conv1d_transpose_1/strided_slice?
5WG_Generator/conv1d_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:27
5WG_Generator/conv1d_transpose_1/strided_slice_1/stack?
7WG_Generator/conv1d_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7WG_Generator/conv1d_transpose_1/strided_slice_1/stack_1?
7WG_Generator/conv1d_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7WG_Generator/conv1d_transpose_1/strided_slice_1/stack_2?
/WG_Generator/conv1d_transpose_1/strided_slice_1StridedSlice.WG_Generator/conv1d_transpose_1/Shape:output:0>WG_Generator/conv1d_transpose_1/strided_slice_1/stack:output:0@WG_Generator/conv1d_transpose_1/strided_slice_1/stack_1:output:0@WG_Generator/conv1d_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask21
/WG_Generator/conv1d_transpose_1/strided_slice_1?
%WG_Generator/conv1d_transpose_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2'
%WG_Generator/conv1d_transpose_1/mul/y?
#WG_Generator/conv1d_transpose_1/mulMul8WG_Generator/conv1d_transpose_1/strided_slice_1:output:0.WG_Generator/conv1d_transpose_1/mul/y:output:0*
T0*
_output_shapes
: 2%
#WG_Generator/conv1d_transpose_1/mul?
'WG_Generator/conv1d_transpose_1/stack/2Const*
_output_shapes
: *
dtype0*
value
B :?2)
'WG_Generator/conv1d_transpose_1/stack/2?
%WG_Generator/conv1d_transpose_1/stackPack6WG_Generator/conv1d_transpose_1/strided_slice:output:0'WG_Generator/conv1d_transpose_1/mul:z:00WG_Generator/conv1d_transpose_1/stack/2:output:0*
N*
T0*
_output_shapes
:2'
%WG_Generator/conv1d_transpose_1/stack?
?WG_Generator/conv1d_transpose_1/conv1d_transpose/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2A
?WG_Generator/conv1d_transpose_1/conv1d_transpose/ExpandDims/dim?
;WG_Generator/conv1d_transpose_1/conv1d_transpose/ExpandDims
ExpandDims2WG_Generator/leaky_re_lu_1/LeakyRelu:activations:0HWG_Generator/conv1d_transpose_1/conv1d_transpose/ExpandDims/dim:output:0*
T0*0
_output_shapes
:?????????@?2=
;WG_Generator/conv1d_transpose_1/conv1d_transpose/ExpandDims?
LWG_Generator/conv1d_transpose_1/conv1d_transpose/ExpandDims_1/ReadVariableOpReadVariableOpUwg_generator_conv1d_transpose_1_conv1d_transpose_expanddims_1_readvariableop_resource*$
_output_shapes
:??*
dtype02N
LWG_Generator/conv1d_transpose_1/conv1d_transpose/ExpandDims_1/ReadVariableOp?
AWG_Generator/conv1d_transpose_1/conv1d_transpose/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2C
AWG_Generator/conv1d_transpose_1/conv1d_transpose/ExpandDims_1/dim?
=WG_Generator/conv1d_transpose_1/conv1d_transpose/ExpandDims_1
ExpandDimsTWG_Generator/conv1d_transpose_1/conv1d_transpose/ExpandDims_1/ReadVariableOp:value:0JWG_Generator/conv1d_transpose_1/conv1d_transpose/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:??2?
=WG_Generator/conv1d_transpose_1/conv1d_transpose/ExpandDims_1?
DWG_Generator/conv1d_transpose_1/conv1d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2F
DWG_Generator/conv1d_transpose_1/conv1d_transpose/strided_slice/stack?
FWG_Generator/conv1d_transpose_1/conv1d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2H
FWG_Generator/conv1d_transpose_1/conv1d_transpose/strided_slice/stack_1?
FWG_Generator/conv1d_transpose_1/conv1d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2H
FWG_Generator/conv1d_transpose_1/conv1d_transpose/strided_slice/stack_2?
>WG_Generator/conv1d_transpose_1/conv1d_transpose/strided_sliceStridedSlice.WG_Generator/conv1d_transpose_1/stack:output:0MWG_Generator/conv1d_transpose_1/conv1d_transpose/strided_slice/stack:output:0OWG_Generator/conv1d_transpose_1/conv1d_transpose/strided_slice/stack_1:output:0OWG_Generator/conv1d_transpose_1/conv1d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2@
>WG_Generator/conv1d_transpose_1/conv1d_transpose/strided_slice?
FWG_Generator/conv1d_transpose_1/conv1d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2H
FWG_Generator/conv1d_transpose_1/conv1d_transpose/strided_slice_1/stack?
HWG_Generator/conv1d_transpose_1/conv1d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2J
HWG_Generator/conv1d_transpose_1/conv1d_transpose/strided_slice_1/stack_1?
HWG_Generator/conv1d_transpose_1/conv1d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2J
HWG_Generator/conv1d_transpose_1/conv1d_transpose/strided_slice_1/stack_2?
@WG_Generator/conv1d_transpose_1/conv1d_transpose/strided_slice_1StridedSlice.WG_Generator/conv1d_transpose_1/stack:output:0OWG_Generator/conv1d_transpose_1/conv1d_transpose/strided_slice_1/stack:output:0QWG_Generator/conv1d_transpose_1/conv1d_transpose/strided_slice_1/stack_1:output:0QWG_Generator/conv1d_transpose_1/conv1d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2B
@WG_Generator/conv1d_transpose_1/conv1d_transpose/strided_slice_1?
@WG_Generator/conv1d_transpose_1/conv1d_transpose/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:2B
@WG_Generator/conv1d_transpose_1/conv1d_transpose/concat/values_1?
<WG_Generator/conv1d_transpose_1/conv1d_transpose/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2>
<WG_Generator/conv1d_transpose_1/conv1d_transpose/concat/axis?
7WG_Generator/conv1d_transpose_1/conv1d_transpose/concatConcatV2GWG_Generator/conv1d_transpose_1/conv1d_transpose/strided_slice:output:0IWG_Generator/conv1d_transpose_1/conv1d_transpose/concat/values_1:output:0IWG_Generator/conv1d_transpose_1/conv1d_transpose/strided_slice_1:output:0EWG_Generator/conv1d_transpose_1/conv1d_transpose/concat/axis:output:0*
N*
T0*
_output_shapes
:29
7WG_Generator/conv1d_transpose_1/conv1d_transpose/concat?
0WG_Generator/conv1d_transpose_1/conv1d_transposeConv2DBackpropInput@WG_Generator/conv1d_transpose_1/conv1d_transpose/concat:output:0FWG_Generator/conv1d_transpose_1/conv1d_transpose/ExpandDims_1:output:0DWG_Generator/conv1d_transpose_1/conv1d_transpose/ExpandDims:output:0*
T0*9
_output_shapes'
%:#???????????????????*
paddingSAME*
strides
22
0WG_Generator/conv1d_transpose_1/conv1d_transpose?
8WG_Generator/conv1d_transpose_1/conv1d_transpose/SqueezeSqueeze9WG_Generator/conv1d_transpose_1/conv1d_transpose:output:0*
T0*-
_output_shapes
:???????????*
squeeze_dims
2:
8WG_Generator/conv1d_transpose_1/conv1d_transpose/Squeeze?
$WG_Generator/leaky_re_lu_2/LeakyRelu	LeakyReluAWG_Generator/conv1d_transpose_1/conv1d_transpose/Squeeze:output:0*-
_output_shapes
:???????????*
alpha%???>2&
$WG_Generator/leaky_re_lu_2/LeakyRelu?
%WG_Generator/conv1d_transpose_2/ShapeShape2WG_Generator/leaky_re_lu_2/LeakyRelu:activations:0*
T0*
_output_shapes
:2'
%WG_Generator/conv1d_transpose_2/Shape?
3WG_Generator/conv1d_transpose_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 25
3WG_Generator/conv1d_transpose_2/strided_slice/stack?
5WG_Generator/conv1d_transpose_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5WG_Generator/conv1d_transpose_2/strided_slice/stack_1?
5WG_Generator/conv1d_transpose_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5WG_Generator/conv1d_transpose_2/strided_slice/stack_2?
-WG_Generator/conv1d_transpose_2/strided_sliceStridedSlice.WG_Generator/conv1d_transpose_2/Shape:output:0<WG_Generator/conv1d_transpose_2/strided_slice/stack:output:0>WG_Generator/conv1d_transpose_2/strided_slice/stack_1:output:0>WG_Generator/conv1d_transpose_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-WG_Generator/conv1d_transpose_2/strided_slice?
5WG_Generator/conv1d_transpose_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:27
5WG_Generator/conv1d_transpose_2/strided_slice_1/stack?
7WG_Generator/conv1d_transpose_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7WG_Generator/conv1d_transpose_2/strided_slice_1/stack_1?
7WG_Generator/conv1d_transpose_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7WG_Generator/conv1d_transpose_2/strided_slice_1/stack_2?
/WG_Generator/conv1d_transpose_2/strided_slice_1StridedSlice.WG_Generator/conv1d_transpose_2/Shape:output:0>WG_Generator/conv1d_transpose_2/strided_slice_1/stack:output:0@WG_Generator/conv1d_transpose_2/strided_slice_1/stack_1:output:0@WG_Generator/conv1d_transpose_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask21
/WG_Generator/conv1d_transpose_2/strided_slice_1?
%WG_Generator/conv1d_transpose_2/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2'
%WG_Generator/conv1d_transpose_2/mul/y?
#WG_Generator/conv1d_transpose_2/mulMul8WG_Generator/conv1d_transpose_2/strided_slice_1:output:0.WG_Generator/conv1d_transpose_2/mul/y:output:0*
T0*
_output_shapes
: 2%
#WG_Generator/conv1d_transpose_2/mul?
'WG_Generator/conv1d_transpose_2/stack/2Const*
_output_shapes
: *
dtype0*
value
B :?2)
'WG_Generator/conv1d_transpose_2/stack/2?
%WG_Generator/conv1d_transpose_2/stackPack6WG_Generator/conv1d_transpose_2/strided_slice:output:0'WG_Generator/conv1d_transpose_2/mul:z:00WG_Generator/conv1d_transpose_2/stack/2:output:0*
N*
T0*
_output_shapes
:2'
%WG_Generator/conv1d_transpose_2/stack?
?WG_Generator/conv1d_transpose_2/conv1d_transpose/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2A
?WG_Generator/conv1d_transpose_2/conv1d_transpose/ExpandDims/dim?
;WG_Generator/conv1d_transpose_2/conv1d_transpose/ExpandDims
ExpandDims2WG_Generator/leaky_re_lu_2/LeakyRelu:activations:0HWG_Generator/conv1d_transpose_2/conv1d_transpose/ExpandDims/dim:output:0*
T0*1
_output_shapes
:???????????2=
;WG_Generator/conv1d_transpose_2/conv1d_transpose/ExpandDims?
LWG_Generator/conv1d_transpose_2/conv1d_transpose/ExpandDims_1/ReadVariableOpReadVariableOpUwg_generator_conv1d_transpose_2_conv1d_transpose_expanddims_1_readvariableop_resource*$
_output_shapes
:??*
dtype02N
LWG_Generator/conv1d_transpose_2/conv1d_transpose/ExpandDims_1/ReadVariableOp?
AWG_Generator/conv1d_transpose_2/conv1d_transpose/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2C
AWG_Generator/conv1d_transpose_2/conv1d_transpose/ExpandDims_1/dim?
=WG_Generator/conv1d_transpose_2/conv1d_transpose/ExpandDims_1
ExpandDimsTWG_Generator/conv1d_transpose_2/conv1d_transpose/ExpandDims_1/ReadVariableOp:value:0JWG_Generator/conv1d_transpose_2/conv1d_transpose/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:??2?
=WG_Generator/conv1d_transpose_2/conv1d_transpose/ExpandDims_1?
DWG_Generator/conv1d_transpose_2/conv1d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2F
DWG_Generator/conv1d_transpose_2/conv1d_transpose/strided_slice/stack?
FWG_Generator/conv1d_transpose_2/conv1d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2H
FWG_Generator/conv1d_transpose_2/conv1d_transpose/strided_slice/stack_1?
FWG_Generator/conv1d_transpose_2/conv1d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2H
FWG_Generator/conv1d_transpose_2/conv1d_transpose/strided_slice/stack_2?
>WG_Generator/conv1d_transpose_2/conv1d_transpose/strided_sliceStridedSlice.WG_Generator/conv1d_transpose_2/stack:output:0MWG_Generator/conv1d_transpose_2/conv1d_transpose/strided_slice/stack:output:0OWG_Generator/conv1d_transpose_2/conv1d_transpose/strided_slice/stack_1:output:0OWG_Generator/conv1d_transpose_2/conv1d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2@
>WG_Generator/conv1d_transpose_2/conv1d_transpose/strided_slice?
FWG_Generator/conv1d_transpose_2/conv1d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2H
FWG_Generator/conv1d_transpose_2/conv1d_transpose/strided_slice_1/stack?
HWG_Generator/conv1d_transpose_2/conv1d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2J
HWG_Generator/conv1d_transpose_2/conv1d_transpose/strided_slice_1/stack_1?
HWG_Generator/conv1d_transpose_2/conv1d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2J
HWG_Generator/conv1d_transpose_2/conv1d_transpose/strided_slice_1/stack_2?
@WG_Generator/conv1d_transpose_2/conv1d_transpose/strided_slice_1StridedSlice.WG_Generator/conv1d_transpose_2/stack:output:0OWG_Generator/conv1d_transpose_2/conv1d_transpose/strided_slice_1/stack:output:0QWG_Generator/conv1d_transpose_2/conv1d_transpose/strided_slice_1/stack_1:output:0QWG_Generator/conv1d_transpose_2/conv1d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2B
@WG_Generator/conv1d_transpose_2/conv1d_transpose/strided_slice_1?
@WG_Generator/conv1d_transpose_2/conv1d_transpose/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:2B
@WG_Generator/conv1d_transpose_2/conv1d_transpose/concat/values_1?
<WG_Generator/conv1d_transpose_2/conv1d_transpose/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2>
<WG_Generator/conv1d_transpose_2/conv1d_transpose/concat/axis?
7WG_Generator/conv1d_transpose_2/conv1d_transpose/concatConcatV2GWG_Generator/conv1d_transpose_2/conv1d_transpose/strided_slice:output:0IWG_Generator/conv1d_transpose_2/conv1d_transpose/concat/values_1:output:0IWG_Generator/conv1d_transpose_2/conv1d_transpose/strided_slice_1:output:0EWG_Generator/conv1d_transpose_2/conv1d_transpose/concat/axis:output:0*
N*
T0*
_output_shapes
:29
7WG_Generator/conv1d_transpose_2/conv1d_transpose/concat?
0WG_Generator/conv1d_transpose_2/conv1d_transposeConv2DBackpropInput@WG_Generator/conv1d_transpose_2/conv1d_transpose/concat:output:0FWG_Generator/conv1d_transpose_2/conv1d_transpose/ExpandDims_1:output:0DWG_Generator/conv1d_transpose_2/conv1d_transpose/ExpandDims:output:0*
T0*9
_output_shapes'
%:#???????????????????*
paddingSAME*
strides
22
0WG_Generator/conv1d_transpose_2/conv1d_transpose?
8WG_Generator/conv1d_transpose_2/conv1d_transpose/SqueezeSqueeze9WG_Generator/conv1d_transpose_2/conv1d_transpose:output:0*
T0*-
_output_shapes
:???????????*
squeeze_dims
2:
8WG_Generator/conv1d_transpose_2/conv1d_transpose/Squeeze?
$WG_Generator/leaky_re_lu_3/LeakyRelu	LeakyReluAWG_Generator/conv1d_transpose_2/conv1d_transpose/Squeeze:output:0*-
_output_shapes
:???????????*
alpha%???>2&
$WG_Generator/leaky_re_lu_3/LeakyRelu?
%WG_Generator/conv1d_transpose_3/ShapeShape2WG_Generator/leaky_re_lu_3/LeakyRelu:activations:0*
T0*
_output_shapes
:2'
%WG_Generator/conv1d_transpose_3/Shape?
3WG_Generator/conv1d_transpose_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 25
3WG_Generator/conv1d_transpose_3/strided_slice/stack?
5WG_Generator/conv1d_transpose_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5WG_Generator/conv1d_transpose_3/strided_slice/stack_1?
5WG_Generator/conv1d_transpose_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5WG_Generator/conv1d_transpose_3/strided_slice/stack_2?
-WG_Generator/conv1d_transpose_3/strided_sliceStridedSlice.WG_Generator/conv1d_transpose_3/Shape:output:0<WG_Generator/conv1d_transpose_3/strided_slice/stack:output:0>WG_Generator/conv1d_transpose_3/strided_slice/stack_1:output:0>WG_Generator/conv1d_transpose_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-WG_Generator/conv1d_transpose_3/strided_slice?
5WG_Generator/conv1d_transpose_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:27
5WG_Generator/conv1d_transpose_3/strided_slice_1/stack?
7WG_Generator/conv1d_transpose_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7WG_Generator/conv1d_transpose_3/strided_slice_1/stack_1?
7WG_Generator/conv1d_transpose_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7WG_Generator/conv1d_transpose_3/strided_slice_1/stack_2?
/WG_Generator/conv1d_transpose_3/strided_slice_1StridedSlice.WG_Generator/conv1d_transpose_3/Shape:output:0>WG_Generator/conv1d_transpose_3/strided_slice_1/stack:output:0@WG_Generator/conv1d_transpose_3/strided_slice_1/stack_1:output:0@WG_Generator/conv1d_transpose_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask21
/WG_Generator/conv1d_transpose_3/strided_slice_1?
%WG_Generator/conv1d_transpose_3/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2'
%WG_Generator/conv1d_transpose_3/mul/y?
#WG_Generator/conv1d_transpose_3/mulMul8WG_Generator/conv1d_transpose_3/strided_slice_1:output:0.WG_Generator/conv1d_transpose_3/mul/y:output:0*
T0*
_output_shapes
: 2%
#WG_Generator/conv1d_transpose_3/mul?
'WG_Generator/conv1d_transpose_3/stack/2Const*
_output_shapes
: *
dtype0*
value	B :@2)
'WG_Generator/conv1d_transpose_3/stack/2?
%WG_Generator/conv1d_transpose_3/stackPack6WG_Generator/conv1d_transpose_3/strided_slice:output:0'WG_Generator/conv1d_transpose_3/mul:z:00WG_Generator/conv1d_transpose_3/stack/2:output:0*
N*
T0*
_output_shapes
:2'
%WG_Generator/conv1d_transpose_3/stack?
?WG_Generator/conv1d_transpose_3/conv1d_transpose/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2A
?WG_Generator/conv1d_transpose_3/conv1d_transpose/ExpandDims/dim?
;WG_Generator/conv1d_transpose_3/conv1d_transpose/ExpandDims
ExpandDims2WG_Generator/leaky_re_lu_3/LeakyRelu:activations:0HWG_Generator/conv1d_transpose_3/conv1d_transpose/ExpandDims/dim:output:0*
T0*1
_output_shapes
:???????????2=
;WG_Generator/conv1d_transpose_3/conv1d_transpose/ExpandDims?
LWG_Generator/conv1d_transpose_3/conv1d_transpose/ExpandDims_1/ReadVariableOpReadVariableOpUwg_generator_conv1d_transpose_3_conv1d_transpose_expanddims_1_readvariableop_resource*#
_output_shapes
:@?*
dtype02N
LWG_Generator/conv1d_transpose_3/conv1d_transpose/ExpandDims_1/ReadVariableOp?
AWG_Generator/conv1d_transpose_3/conv1d_transpose/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2C
AWG_Generator/conv1d_transpose_3/conv1d_transpose/ExpandDims_1/dim?
=WG_Generator/conv1d_transpose_3/conv1d_transpose/ExpandDims_1
ExpandDimsTWG_Generator/conv1d_transpose_3/conv1d_transpose/ExpandDims_1/ReadVariableOp:value:0JWG_Generator/conv1d_transpose_3/conv1d_transpose/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:@?2?
=WG_Generator/conv1d_transpose_3/conv1d_transpose/ExpandDims_1?
DWG_Generator/conv1d_transpose_3/conv1d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2F
DWG_Generator/conv1d_transpose_3/conv1d_transpose/strided_slice/stack?
FWG_Generator/conv1d_transpose_3/conv1d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2H
FWG_Generator/conv1d_transpose_3/conv1d_transpose/strided_slice/stack_1?
FWG_Generator/conv1d_transpose_3/conv1d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2H
FWG_Generator/conv1d_transpose_3/conv1d_transpose/strided_slice/stack_2?
>WG_Generator/conv1d_transpose_3/conv1d_transpose/strided_sliceStridedSlice.WG_Generator/conv1d_transpose_3/stack:output:0MWG_Generator/conv1d_transpose_3/conv1d_transpose/strided_slice/stack:output:0OWG_Generator/conv1d_transpose_3/conv1d_transpose/strided_slice/stack_1:output:0OWG_Generator/conv1d_transpose_3/conv1d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2@
>WG_Generator/conv1d_transpose_3/conv1d_transpose/strided_slice?
FWG_Generator/conv1d_transpose_3/conv1d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2H
FWG_Generator/conv1d_transpose_3/conv1d_transpose/strided_slice_1/stack?
HWG_Generator/conv1d_transpose_3/conv1d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2J
HWG_Generator/conv1d_transpose_3/conv1d_transpose/strided_slice_1/stack_1?
HWG_Generator/conv1d_transpose_3/conv1d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2J
HWG_Generator/conv1d_transpose_3/conv1d_transpose/strided_slice_1/stack_2?
@WG_Generator/conv1d_transpose_3/conv1d_transpose/strided_slice_1StridedSlice.WG_Generator/conv1d_transpose_3/stack:output:0OWG_Generator/conv1d_transpose_3/conv1d_transpose/strided_slice_1/stack:output:0QWG_Generator/conv1d_transpose_3/conv1d_transpose/strided_slice_1/stack_1:output:0QWG_Generator/conv1d_transpose_3/conv1d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2B
@WG_Generator/conv1d_transpose_3/conv1d_transpose/strided_slice_1?
@WG_Generator/conv1d_transpose_3/conv1d_transpose/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:2B
@WG_Generator/conv1d_transpose_3/conv1d_transpose/concat/values_1?
<WG_Generator/conv1d_transpose_3/conv1d_transpose/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2>
<WG_Generator/conv1d_transpose_3/conv1d_transpose/concat/axis?
7WG_Generator/conv1d_transpose_3/conv1d_transpose/concatConcatV2GWG_Generator/conv1d_transpose_3/conv1d_transpose/strided_slice:output:0IWG_Generator/conv1d_transpose_3/conv1d_transpose/concat/values_1:output:0IWG_Generator/conv1d_transpose_3/conv1d_transpose/strided_slice_1:output:0EWG_Generator/conv1d_transpose_3/conv1d_transpose/concat/axis:output:0*
N*
T0*
_output_shapes
:29
7WG_Generator/conv1d_transpose_3/conv1d_transpose/concat?
0WG_Generator/conv1d_transpose_3/conv1d_transposeConv2DBackpropInput@WG_Generator/conv1d_transpose_3/conv1d_transpose/concat:output:0FWG_Generator/conv1d_transpose_3/conv1d_transpose/ExpandDims_1:output:0DWG_Generator/conv1d_transpose_3/conv1d_transpose/ExpandDims:output:0*
T0*8
_output_shapes&
$:"??????????????????@*
paddingSAME*
strides
22
0WG_Generator/conv1d_transpose_3/conv1d_transpose?
8WG_Generator/conv1d_transpose_3/conv1d_transpose/SqueezeSqueeze9WG_Generator/conv1d_transpose_3/conv1d_transpose:output:0*
T0*,
_output_shapes
:?????????? @*
squeeze_dims
2:
8WG_Generator/conv1d_transpose_3/conv1d_transpose/Squeeze?
$WG_Generator/leaky_re_lu_4/LeakyRelu	LeakyReluAWG_Generator/conv1d_transpose_3/conv1d_transpose/Squeeze:output:0*,
_output_shapes
:?????????? @*
alpha%???>2&
$WG_Generator/leaky_re_lu_4/LeakyRelu?
%WG_Generator/conv1d_transpose_4/ShapeShape2WG_Generator/leaky_re_lu_4/LeakyRelu:activations:0*
T0*
_output_shapes
:2'
%WG_Generator/conv1d_transpose_4/Shape?
3WG_Generator/conv1d_transpose_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 25
3WG_Generator/conv1d_transpose_4/strided_slice/stack?
5WG_Generator/conv1d_transpose_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5WG_Generator/conv1d_transpose_4/strided_slice/stack_1?
5WG_Generator/conv1d_transpose_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5WG_Generator/conv1d_transpose_4/strided_slice/stack_2?
-WG_Generator/conv1d_transpose_4/strided_sliceStridedSlice.WG_Generator/conv1d_transpose_4/Shape:output:0<WG_Generator/conv1d_transpose_4/strided_slice/stack:output:0>WG_Generator/conv1d_transpose_4/strided_slice/stack_1:output:0>WG_Generator/conv1d_transpose_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-WG_Generator/conv1d_transpose_4/strided_slice?
5WG_Generator/conv1d_transpose_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:27
5WG_Generator/conv1d_transpose_4/strided_slice_1/stack?
7WG_Generator/conv1d_transpose_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7WG_Generator/conv1d_transpose_4/strided_slice_1/stack_1?
7WG_Generator/conv1d_transpose_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7WG_Generator/conv1d_transpose_4/strided_slice_1/stack_2?
/WG_Generator/conv1d_transpose_4/strided_slice_1StridedSlice.WG_Generator/conv1d_transpose_4/Shape:output:0>WG_Generator/conv1d_transpose_4/strided_slice_1/stack:output:0@WG_Generator/conv1d_transpose_4/strided_slice_1/stack_1:output:0@WG_Generator/conv1d_transpose_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask21
/WG_Generator/conv1d_transpose_4/strided_slice_1?
%WG_Generator/conv1d_transpose_4/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2'
%WG_Generator/conv1d_transpose_4/mul/y?
#WG_Generator/conv1d_transpose_4/mulMul8WG_Generator/conv1d_transpose_4/strided_slice_1:output:0.WG_Generator/conv1d_transpose_4/mul/y:output:0*
T0*
_output_shapes
: 2%
#WG_Generator/conv1d_transpose_4/mul?
'WG_Generator/conv1d_transpose_4/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2)
'WG_Generator/conv1d_transpose_4/stack/2?
%WG_Generator/conv1d_transpose_4/stackPack6WG_Generator/conv1d_transpose_4/strided_slice:output:0'WG_Generator/conv1d_transpose_4/mul:z:00WG_Generator/conv1d_transpose_4/stack/2:output:0*
N*
T0*
_output_shapes
:2'
%WG_Generator/conv1d_transpose_4/stack?
?WG_Generator/conv1d_transpose_4/conv1d_transpose/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2A
?WG_Generator/conv1d_transpose_4/conv1d_transpose/ExpandDims/dim?
;WG_Generator/conv1d_transpose_4/conv1d_transpose/ExpandDims
ExpandDims2WG_Generator/leaky_re_lu_4/LeakyRelu:activations:0HWG_Generator/conv1d_transpose_4/conv1d_transpose/ExpandDims/dim:output:0*
T0*0
_output_shapes
:?????????? @2=
;WG_Generator/conv1d_transpose_4/conv1d_transpose/ExpandDims?
LWG_Generator/conv1d_transpose_4/conv1d_transpose/ExpandDims_1/ReadVariableOpReadVariableOpUwg_generator_conv1d_transpose_4_conv1d_transpose_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02N
LWG_Generator/conv1d_transpose_4/conv1d_transpose/ExpandDims_1/ReadVariableOp?
AWG_Generator/conv1d_transpose_4/conv1d_transpose/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2C
AWG_Generator/conv1d_transpose_4/conv1d_transpose/ExpandDims_1/dim?
=WG_Generator/conv1d_transpose_4/conv1d_transpose/ExpandDims_1
ExpandDimsTWG_Generator/conv1d_transpose_4/conv1d_transpose/ExpandDims_1/ReadVariableOp:value:0JWG_Generator/conv1d_transpose_4/conv1d_transpose/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@2?
=WG_Generator/conv1d_transpose_4/conv1d_transpose/ExpandDims_1?
DWG_Generator/conv1d_transpose_4/conv1d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2F
DWG_Generator/conv1d_transpose_4/conv1d_transpose/strided_slice/stack?
FWG_Generator/conv1d_transpose_4/conv1d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2H
FWG_Generator/conv1d_transpose_4/conv1d_transpose/strided_slice/stack_1?
FWG_Generator/conv1d_transpose_4/conv1d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2H
FWG_Generator/conv1d_transpose_4/conv1d_transpose/strided_slice/stack_2?
>WG_Generator/conv1d_transpose_4/conv1d_transpose/strided_sliceStridedSlice.WG_Generator/conv1d_transpose_4/stack:output:0MWG_Generator/conv1d_transpose_4/conv1d_transpose/strided_slice/stack:output:0OWG_Generator/conv1d_transpose_4/conv1d_transpose/strided_slice/stack_1:output:0OWG_Generator/conv1d_transpose_4/conv1d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2@
>WG_Generator/conv1d_transpose_4/conv1d_transpose/strided_slice?
FWG_Generator/conv1d_transpose_4/conv1d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2H
FWG_Generator/conv1d_transpose_4/conv1d_transpose/strided_slice_1/stack?
HWG_Generator/conv1d_transpose_4/conv1d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2J
HWG_Generator/conv1d_transpose_4/conv1d_transpose/strided_slice_1/stack_1?
HWG_Generator/conv1d_transpose_4/conv1d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2J
HWG_Generator/conv1d_transpose_4/conv1d_transpose/strided_slice_1/stack_2?
@WG_Generator/conv1d_transpose_4/conv1d_transpose/strided_slice_1StridedSlice.WG_Generator/conv1d_transpose_4/stack:output:0OWG_Generator/conv1d_transpose_4/conv1d_transpose/strided_slice_1/stack:output:0QWG_Generator/conv1d_transpose_4/conv1d_transpose/strided_slice_1/stack_1:output:0QWG_Generator/conv1d_transpose_4/conv1d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2B
@WG_Generator/conv1d_transpose_4/conv1d_transpose/strided_slice_1?
@WG_Generator/conv1d_transpose_4/conv1d_transpose/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:2B
@WG_Generator/conv1d_transpose_4/conv1d_transpose/concat/values_1?
<WG_Generator/conv1d_transpose_4/conv1d_transpose/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2>
<WG_Generator/conv1d_transpose_4/conv1d_transpose/concat/axis?
7WG_Generator/conv1d_transpose_4/conv1d_transpose/concatConcatV2GWG_Generator/conv1d_transpose_4/conv1d_transpose/strided_slice:output:0IWG_Generator/conv1d_transpose_4/conv1d_transpose/concat/values_1:output:0IWG_Generator/conv1d_transpose_4/conv1d_transpose/strided_slice_1:output:0EWG_Generator/conv1d_transpose_4/conv1d_transpose/concat/axis:output:0*
N*
T0*
_output_shapes
:29
7WG_Generator/conv1d_transpose_4/conv1d_transpose/concat?
0WG_Generator/conv1d_transpose_4/conv1d_transposeConv2DBackpropInput@WG_Generator/conv1d_transpose_4/conv1d_transpose/concat:output:0FWG_Generator/conv1d_transpose_4/conv1d_transpose/ExpandDims_1:output:0DWG_Generator/conv1d_transpose_4/conv1d_transpose/ExpandDims:output:0*
T0*8
_output_shapes&
$:"??????????????????*
paddingSAME*
strides
22
0WG_Generator/conv1d_transpose_4/conv1d_transpose?
8WG_Generator/conv1d_transpose_4/conv1d_transpose/SqueezeSqueeze9WG_Generator/conv1d_transpose_4/conv1d_transpose:output:0*
T0*-
_output_shapes
:???????????*
squeeze_dims
2:
8WG_Generator/conv1d_transpose_4/conv1d_transpose/Squeeze?
$WG_Generator/conv1d_transpose_4/TanhTanhAWG_Generator/conv1d_transpose_4/conv1d_transpose/Squeeze:output:0*
T0*-
_output_shapes
:???????????2&
$WG_Generator/conv1d_transpose_4/Tanh?
IdentityIdentity(WG_Generator/conv1d_transpose_4/Tanh:y:0^NoOp*
T0*-
_output_shapes
:???????????2

Identity?
NoOpNoOpK^WG_Generator/conv1d_transpose/conv1d_transpose/ExpandDims_1/ReadVariableOpM^WG_Generator/conv1d_transpose_1/conv1d_transpose/ExpandDims_1/ReadVariableOpM^WG_Generator/conv1d_transpose_2/conv1d_transpose/ExpandDims_1/ReadVariableOpM^WG_Generator/conv1d_transpose_3/conv1d_transpose/ExpandDims_1/ReadVariableOpM^WG_Generator/conv1d_transpose_4/conv1d_transpose/ExpandDims_1/ReadVariableOp)^WG_Generator/dense/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : : : : : 2?
JWG_Generator/conv1d_transpose/conv1d_transpose/ExpandDims_1/ReadVariableOpJWG_Generator/conv1d_transpose/conv1d_transpose/ExpandDims_1/ReadVariableOp2?
LWG_Generator/conv1d_transpose_1/conv1d_transpose/ExpandDims_1/ReadVariableOpLWG_Generator/conv1d_transpose_1/conv1d_transpose/ExpandDims_1/ReadVariableOp2?
LWG_Generator/conv1d_transpose_2/conv1d_transpose/ExpandDims_1/ReadVariableOpLWG_Generator/conv1d_transpose_2/conv1d_transpose/ExpandDims_1/ReadVariableOp2?
LWG_Generator/conv1d_transpose_3/conv1d_transpose/ExpandDims_1/ReadVariableOpLWG_Generator/conv1d_transpose_3/conv1d_transpose/ExpandDims_1/ReadVariableOp2?
LWG_Generator/conv1d_transpose_4/conv1d_transpose/ExpandDims_1/ReadVariableOpLWG_Generator/conv1d_transpose_4/conv1d_transpose/ExpandDims_1/ReadVariableOp2T
(WG_Generator/dense/MatMul/ReadVariableOp(WG_Generator/dense/MatMul/ReadVariableOp:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1
?
?
C__inference_dense_layer_call_and_return_conditional_losses_78011249

inputs2
matmul_readvariableop_resource:
??
identity??MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpu
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*)
_output_shapes
:???????????2
MatMulm
IdentityIdentityMatMul:product:0^NoOp*
T0*)
_output_shapes
:???????????2

Identityf
NoOpNoOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
/__inference_WG_Generator_layer_call_fn_78012129

inputs
unknown:
??!
	unknown_0:??!
	unknown_1:??!
	unknown_2:?? 
	unknown_3:@?
	unknown_4:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:???????????*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_WG_Generator_layer_call_and_return_conditional_losses_780115002
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*-
_output_shapes
:???????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
3__inference_conv1d_transpose_layer_call_fn_78012267

inputs
unknown:??
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:???????????????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_conv1d_transpose_layer_call_and_return_conditional_losses_780108332
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:???????????????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:???????????????????: 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:???????????????????
 
_user_specified_nameinputs
?.
?
P__inference_conv1d_transpose_1_layer_call_and_return_conditional_losses_78012356

inputsM
5conv1d_transpose_expanddims_1_readvariableop_resource:??
identity??,conv1d_transpose/ExpandDims_1/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulU
stack/2Const*
_output_shapes
: *
dtype0*
value
B :?2	
stack/2w
stackPackstrided_slice:output:0mul:z:0stack/2:output:0*
N*
T0*
_output_shapes
:2
stack?
conv1d_transpose/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2!
conv1d_transpose/ExpandDims/dim?
conv1d_transpose/ExpandDims
ExpandDimsinputs(conv1d_transpose/ExpandDims/dim:output:0*
T0*0
_output_shapes
:?????????@?2
conv1d_transpose/ExpandDims?
,conv1d_transpose/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_transpose_expanddims_1_readvariableop_resource*$
_output_shapes
:??*
dtype02.
,conv1d_transpose/ExpandDims_1/ReadVariableOp?
!conv1d_transpose/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!conv1d_transpose/ExpandDims_1/dim?
conv1d_transpose/ExpandDims_1
ExpandDims4conv1d_transpose/ExpandDims_1/ReadVariableOp:value:0*conv1d_transpose/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:??2
conv1d_transpose/ExpandDims_1?
$conv1d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$conv1d_transpose/strided_slice/stack?
&conv1d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv1d_transpose/strided_slice/stack_1?
&conv1d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv1d_transpose/strided_slice/stack_2?
conv1d_transpose/strided_sliceStridedSlicestack:output:0-conv1d_transpose/strided_slice/stack:output:0/conv1d_transpose/strided_slice/stack_1:output:0/conv1d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2 
conv1d_transpose/strided_slice?
&conv1d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2(
&conv1d_transpose/strided_slice_1/stack?
(conv1d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(conv1d_transpose/strided_slice_1/stack_1?
(conv1d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv1d_transpose/strided_slice_1/stack_2?
 conv1d_transpose/strided_slice_1StridedSlicestack:output:0/conv1d_transpose/strided_slice_1/stack:output:01conv1d_transpose/strided_slice_1/stack_1:output:01conv1d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2"
 conv1d_transpose/strided_slice_1?
 conv1d_transpose/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 conv1d_transpose/concat/values_1~
conv1d_transpose/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d_transpose/concat/axis?
conv1d_transpose/concatConcatV2'conv1d_transpose/strided_slice:output:0)conv1d_transpose/concat/values_1:output:0)conv1d_transpose/strided_slice_1:output:0%conv1d_transpose/concat/axis:output:0*
N*
T0*
_output_shapes
:2
conv1d_transpose/concat?
conv1d_transposeConv2DBackpropInput conv1d_transpose/concat:output:0&conv1d_transpose/ExpandDims_1:output:0$conv1d_transpose/ExpandDims:output:0*
T0*9
_output_shapes'
%:#???????????????????*
paddingSAME*
strides
2
conv1d_transpose?
conv1d_transpose/SqueezeSqueezeconv1d_transpose:output:0*
T0*-
_output_shapes
:???????????*
squeeze_dims
2
conv1d_transpose/Squeeze?
IdentityIdentity!conv1d_transpose/Squeeze:output:0^NoOp*
T0*-
_output_shapes
:???????????2

Identity}
NoOpNoOp-^conv1d_transpose/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*-
_input_shapes
:?????????@?: 2\
,conv1d_transpose/ExpandDims_1/ReadVariableOp,conv1d_transpose/ExpandDims_1/ReadVariableOp:T P
,
_output_shapes
:?????????@?
 
_user_specified_nameinputs
?
g
K__inference_leaky_re_lu_3_layer_call_and_return_conditional_losses_78011411

inputs
identityj
	LeakyRelu	LeakyReluinputs*-
_output_shapes
:???????????*
alpha%???>2
	LeakyReluq
IdentityIdentityLeakyRelu:activations:0*
T0*-
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:???????????:U Q
-
_output_shapes
:???????????
 
_user_specified_nameinputs
ҹ
?
J__inference_WG_Generator_layer_call_and_return_conditional_losses_78011930

inputs8
$dense_matmul_readvariableop_resource:
??^
Fconv1d_transpose_conv1d_transpose_expanddims_1_readvariableop_resource:??`
Hconv1d_transpose_1_conv1d_transpose_expanddims_1_readvariableop_resource:??`
Hconv1d_transpose_2_conv1d_transpose_expanddims_1_readvariableop_resource:??_
Hconv1d_transpose_3_conv1d_transpose_expanddims_1_readvariableop_resource:@?^
Hconv1d_transpose_4_conv1d_transpose_expanddims_1_readvariableop_resource:@
identity??=conv1d_transpose/conv1d_transpose/ExpandDims_1/ReadVariableOp??conv1d_transpose_1/conv1d_transpose/ExpandDims_1/ReadVariableOp??conv1d_transpose_2/conv1d_transpose/ExpandDims_1/ReadVariableOp??conv1d_transpose_3/conv1d_transpose/ExpandDims_1/ReadVariableOp??conv1d_transpose_4/conv1d_transpose/ExpandDims_1/ReadVariableOp?dense/MatMul/ReadVariableOp?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
dense/MatMul/ReadVariableOp?
dense/MatMulMatMulinputs#dense/MatMul/ReadVariableOp:value:0*
T0*)
_output_shapes
:???????????2
dense/MatMul?
leaky_re_lu/LeakyRelu	LeakyReludense/MatMul:product:0*)
_output_shapes
:???????????*
alpha%???>2
leaky_re_lu/LeakyReluq
reshape/ShapeShape#leaky_re_lu/LeakyRelu:activations:0*
T0*
_output_shapes
:2
reshape/Shape?
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape/strided_slice/stack?
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_1?
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_2?
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape/strided_slicet
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/1u
reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value
B :?2
reshape/Reshape/shape/2?
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
reshape/Reshape/shape?
reshape/ReshapeReshape#leaky_re_lu/LeakyRelu:activations:0reshape/Reshape/shape:output:0*
T0*,
_output_shapes
:??????????2
reshape/Reshapex
conv1d_transpose/ShapeShapereshape/Reshape:output:0*
T0*
_output_shapes
:2
conv1d_transpose/Shape?
$conv1d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$conv1d_transpose/strided_slice/stack?
&conv1d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv1d_transpose/strided_slice/stack_1?
&conv1d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv1d_transpose/strided_slice/stack_2?
conv1d_transpose/strided_sliceStridedSliceconv1d_transpose/Shape:output:0-conv1d_transpose/strided_slice/stack:output:0/conv1d_transpose/strided_slice/stack_1:output:0/conv1d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
conv1d_transpose/strided_slice?
&conv1d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2(
&conv1d_transpose/strided_slice_1/stack?
(conv1d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv1d_transpose/strided_slice_1/stack_1?
(conv1d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv1d_transpose/strided_slice_1/stack_2?
 conv1d_transpose/strided_slice_1StridedSliceconv1d_transpose/Shape:output:0/conv1d_transpose/strided_slice_1/stack:output:01conv1d_transpose/strided_slice_1/stack_1:output:01conv1d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv1d_transpose/strided_slice_1r
conv1d_transpose/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
conv1d_transpose/mul/y?
conv1d_transpose/mulMul)conv1d_transpose/strided_slice_1:output:0conv1d_transpose/mul/y:output:0*
T0*
_output_shapes
: 2
conv1d_transpose/mulw
conv1d_transpose/stack/2Const*
_output_shapes
: *
dtype0*
value
B :?2
conv1d_transpose/stack/2?
conv1d_transpose/stackPack'conv1d_transpose/strided_slice:output:0conv1d_transpose/mul:z:0!conv1d_transpose/stack/2:output:0*
N*
T0*
_output_shapes
:2
conv1d_transpose/stack?
0conv1d_transpose/conv1d_transpose/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :22
0conv1d_transpose/conv1d_transpose/ExpandDims/dim?
,conv1d_transpose/conv1d_transpose/ExpandDims
ExpandDimsreshape/Reshape:output:09conv1d_transpose/conv1d_transpose/ExpandDims/dim:output:0*
T0*0
_output_shapes
:??????????2.
,conv1d_transpose/conv1d_transpose/ExpandDims?
=conv1d_transpose/conv1d_transpose/ExpandDims_1/ReadVariableOpReadVariableOpFconv1d_transpose_conv1d_transpose_expanddims_1_readvariableop_resource*$
_output_shapes
:??*
dtype02?
=conv1d_transpose/conv1d_transpose/ExpandDims_1/ReadVariableOp?
2conv1d_transpose/conv1d_transpose/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 24
2conv1d_transpose/conv1d_transpose/ExpandDims_1/dim?
.conv1d_transpose/conv1d_transpose/ExpandDims_1
ExpandDimsEconv1d_transpose/conv1d_transpose/ExpandDims_1/ReadVariableOp:value:0;conv1d_transpose/conv1d_transpose/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:??20
.conv1d_transpose/conv1d_transpose/ExpandDims_1?
5conv1d_transpose/conv1d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 27
5conv1d_transpose/conv1d_transpose/strided_slice/stack?
7conv1d_transpose/conv1d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7conv1d_transpose/conv1d_transpose/strided_slice/stack_1?
7conv1d_transpose/conv1d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7conv1d_transpose/conv1d_transpose/strided_slice/stack_2?
/conv1d_transpose/conv1d_transpose/strided_sliceStridedSliceconv1d_transpose/stack:output:0>conv1d_transpose/conv1d_transpose/strided_slice/stack:output:0@conv1d_transpose/conv1d_transpose/strided_slice/stack_1:output:0@conv1d_transpose/conv1d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask21
/conv1d_transpose/conv1d_transpose/strided_slice?
7conv1d_transpose/conv1d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:29
7conv1d_transpose/conv1d_transpose/strided_slice_1/stack?
9conv1d_transpose/conv1d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2;
9conv1d_transpose/conv1d_transpose/strided_slice_1/stack_1?
9conv1d_transpose/conv1d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2;
9conv1d_transpose/conv1d_transpose/strided_slice_1/stack_2?
1conv1d_transpose/conv1d_transpose/strided_slice_1StridedSliceconv1d_transpose/stack:output:0@conv1d_transpose/conv1d_transpose/strided_slice_1/stack:output:0Bconv1d_transpose/conv1d_transpose/strided_slice_1/stack_1:output:0Bconv1d_transpose/conv1d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask23
1conv1d_transpose/conv1d_transpose/strided_slice_1?
1conv1d_transpose/conv1d_transpose/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:23
1conv1d_transpose/conv1d_transpose/concat/values_1?
-conv1d_transpose/conv1d_transpose/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-conv1d_transpose/conv1d_transpose/concat/axis?
(conv1d_transpose/conv1d_transpose/concatConcatV28conv1d_transpose/conv1d_transpose/strided_slice:output:0:conv1d_transpose/conv1d_transpose/concat/values_1:output:0:conv1d_transpose/conv1d_transpose/strided_slice_1:output:06conv1d_transpose/conv1d_transpose/concat/axis:output:0*
N*
T0*
_output_shapes
:2*
(conv1d_transpose/conv1d_transpose/concat?
!conv1d_transpose/conv1d_transposeConv2DBackpropInput1conv1d_transpose/conv1d_transpose/concat:output:07conv1d_transpose/conv1d_transpose/ExpandDims_1:output:05conv1d_transpose/conv1d_transpose/ExpandDims:output:0*
T0*9
_output_shapes'
%:#???????????????????*
paddingSAME*
strides
2#
!conv1d_transpose/conv1d_transpose?
)conv1d_transpose/conv1d_transpose/SqueezeSqueeze*conv1d_transpose/conv1d_transpose:output:0*
T0*,
_output_shapes
:?????????@?*
squeeze_dims
2+
)conv1d_transpose/conv1d_transpose/Squeeze?
leaky_re_lu_1/LeakyRelu	LeakyRelu2conv1d_transpose/conv1d_transpose/Squeeze:output:0*,
_output_shapes
:?????????@?*
alpha%???>2
leaky_re_lu_1/LeakyRelu?
conv1d_transpose_1/ShapeShape%leaky_re_lu_1/LeakyRelu:activations:0*
T0*
_output_shapes
:2
conv1d_transpose_1/Shape?
&conv1d_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv1d_transpose_1/strided_slice/stack?
(conv1d_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv1d_transpose_1/strided_slice/stack_1?
(conv1d_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv1d_transpose_1/strided_slice/stack_2?
 conv1d_transpose_1/strided_sliceStridedSlice!conv1d_transpose_1/Shape:output:0/conv1d_transpose_1/strided_slice/stack:output:01conv1d_transpose_1/strided_slice/stack_1:output:01conv1d_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv1d_transpose_1/strided_slice?
(conv1d_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2*
(conv1d_transpose_1/strided_slice_1/stack?
*conv1d_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv1d_transpose_1/strided_slice_1/stack_1?
*conv1d_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv1d_transpose_1/strided_slice_1/stack_2?
"conv1d_transpose_1/strided_slice_1StridedSlice!conv1d_transpose_1/Shape:output:01conv1d_transpose_1/strided_slice_1/stack:output:03conv1d_transpose_1/strided_slice_1/stack_1:output:03conv1d_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv1d_transpose_1/strided_slice_1v
conv1d_transpose_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
conv1d_transpose_1/mul/y?
conv1d_transpose_1/mulMul+conv1d_transpose_1/strided_slice_1:output:0!conv1d_transpose_1/mul/y:output:0*
T0*
_output_shapes
: 2
conv1d_transpose_1/mul{
conv1d_transpose_1/stack/2Const*
_output_shapes
: *
dtype0*
value
B :?2
conv1d_transpose_1/stack/2?
conv1d_transpose_1/stackPack)conv1d_transpose_1/strided_slice:output:0conv1d_transpose_1/mul:z:0#conv1d_transpose_1/stack/2:output:0*
N*
T0*
_output_shapes
:2
conv1d_transpose_1/stack?
2conv1d_transpose_1/conv1d_transpose/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :24
2conv1d_transpose_1/conv1d_transpose/ExpandDims/dim?
.conv1d_transpose_1/conv1d_transpose/ExpandDims
ExpandDims%leaky_re_lu_1/LeakyRelu:activations:0;conv1d_transpose_1/conv1d_transpose/ExpandDims/dim:output:0*
T0*0
_output_shapes
:?????????@?20
.conv1d_transpose_1/conv1d_transpose/ExpandDims?
?conv1d_transpose_1/conv1d_transpose/ExpandDims_1/ReadVariableOpReadVariableOpHconv1d_transpose_1_conv1d_transpose_expanddims_1_readvariableop_resource*$
_output_shapes
:??*
dtype02A
?conv1d_transpose_1/conv1d_transpose/ExpandDims_1/ReadVariableOp?
4conv1d_transpose_1/conv1d_transpose/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 26
4conv1d_transpose_1/conv1d_transpose/ExpandDims_1/dim?
0conv1d_transpose_1/conv1d_transpose/ExpandDims_1
ExpandDimsGconv1d_transpose_1/conv1d_transpose/ExpandDims_1/ReadVariableOp:value:0=conv1d_transpose_1/conv1d_transpose/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:??22
0conv1d_transpose_1/conv1d_transpose/ExpandDims_1?
7conv1d_transpose_1/conv1d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 29
7conv1d_transpose_1/conv1d_transpose/strided_slice/stack?
9conv1d_transpose_1/conv1d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2;
9conv1d_transpose_1/conv1d_transpose/strided_slice/stack_1?
9conv1d_transpose_1/conv1d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2;
9conv1d_transpose_1/conv1d_transpose/strided_slice/stack_2?
1conv1d_transpose_1/conv1d_transpose/strided_sliceStridedSlice!conv1d_transpose_1/stack:output:0@conv1d_transpose_1/conv1d_transpose/strided_slice/stack:output:0Bconv1d_transpose_1/conv1d_transpose/strided_slice/stack_1:output:0Bconv1d_transpose_1/conv1d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask23
1conv1d_transpose_1/conv1d_transpose/strided_slice?
9conv1d_transpose_1/conv1d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2;
9conv1d_transpose_1/conv1d_transpose/strided_slice_1/stack?
;conv1d_transpose_1/conv1d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2=
;conv1d_transpose_1/conv1d_transpose/strided_slice_1/stack_1?
;conv1d_transpose_1/conv1d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2=
;conv1d_transpose_1/conv1d_transpose/strided_slice_1/stack_2?
3conv1d_transpose_1/conv1d_transpose/strided_slice_1StridedSlice!conv1d_transpose_1/stack:output:0Bconv1d_transpose_1/conv1d_transpose/strided_slice_1/stack:output:0Dconv1d_transpose_1/conv1d_transpose/strided_slice_1/stack_1:output:0Dconv1d_transpose_1/conv1d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask25
3conv1d_transpose_1/conv1d_transpose/strided_slice_1?
3conv1d_transpose_1/conv1d_transpose/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:25
3conv1d_transpose_1/conv1d_transpose/concat/values_1?
/conv1d_transpose_1/conv1d_transpose/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/conv1d_transpose_1/conv1d_transpose/concat/axis?
*conv1d_transpose_1/conv1d_transpose/concatConcatV2:conv1d_transpose_1/conv1d_transpose/strided_slice:output:0<conv1d_transpose_1/conv1d_transpose/concat/values_1:output:0<conv1d_transpose_1/conv1d_transpose/strided_slice_1:output:08conv1d_transpose_1/conv1d_transpose/concat/axis:output:0*
N*
T0*
_output_shapes
:2,
*conv1d_transpose_1/conv1d_transpose/concat?
#conv1d_transpose_1/conv1d_transposeConv2DBackpropInput3conv1d_transpose_1/conv1d_transpose/concat:output:09conv1d_transpose_1/conv1d_transpose/ExpandDims_1:output:07conv1d_transpose_1/conv1d_transpose/ExpandDims:output:0*
T0*9
_output_shapes'
%:#???????????????????*
paddingSAME*
strides
2%
#conv1d_transpose_1/conv1d_transpose?
+conv1d_transpose_1/conv1d_transpose/SqueezeSqueeze,conv1d_transpose_1/conv1d_transpose:output:0*
T0*-
_output_shapes
:???????????*
squeeze_dims
2-
+conv1d_transpose_1/conv1d_transpose/Squeeze?
leaky_re_lu_2/LeakyRelu	LeakyRelu4conv1d_transpose_1/conv1d_transpose/Squeeze:output:0*-
_output_shapes
:???????????*
alpha%???>2
leaky_re_lu_2/LeakyRelu?
conv1d_transpose_2/ShapeShape%leaky_re_lu_2/LeakyRelu:activations:0*
T0*
_output_shapes
:2
conv1d_transpose_2/Shape?
&conv1d_transpose_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv1d_transpose_2/strided_slice/stack?
(conv1d_transpose_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv1d_transpose_2/strided_slice/stack_1?
(conv1d_transpose_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv1d_transpose_2/strided_slice/stack_2?
 conv1d_transpose_2/strided_sliceStridedSlice!conv1d_transpose_2/Shape:output:0/conv1d_transpose_2/strided_slice/stack:output:01conv1d_transpose_2/strided_slice/stack_1:output:01conv1d_transpose_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv1d_transpose_2/strided_slice?
(conv1d_transpose_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2*
(conv1d_transpose_2/strided_slice_1/stack?
*conv1d_transpose_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv1d_transpose_2/strided_slice_1/stack_1?
*conv1d_transpose_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv1d_transpose_2/strided_slice_1/stack_2?
"conv1d_transpose_2/strided_slice_1StridedSlice!conv1d_transpose_2/Shape:output:01conv1d_transpose_2/strided_slice_1/stack:output:03conv1d_transpose_2/strided_slice_1/stack_1:output:03conv1d_transpose_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv1d_transpose_2/strided_slice_1v
conv1d_transpose_2/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
conv1d_transpose_2/mul/y?
conv1d_transpose_2/mulMul+conv1d_transpose_2/strided_slice_1:output:0!conv1d_transpose_2/mul/y:output:0*
T0*
_output_shapes
: 2
conv1d_transpose_2/mul{
conv1d_transpose_2/stack/2Const*
_output_shapes
: *
dtype0*
value
B :?2
conv1d_transpose_2/stack/2?
conv1d_transpose_2/stackPack)conv1d_transpose_2/strided_slice:output:0conv1d_transpose_2/mul:z:0#conv1d_transpose_2/stack/2:output:0*
N*
T0*
_output_shapes
:2
conv1d_transpose_2/stack?
2conv1d_transpose_2/conv1d_transpose/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :24
2conv1d_transpose_2/conv1d_transpose/ExpandDims/dim?
.conv1d_transpose_2/conv1d_transpose/ExpandDims
ExpandDims%leaky_re_lu_2/LeakyRelu:activations:0;conv1d_transpose_2/conv1d_transpose/ExpandDims/dim:output:0*
T0*1
_output_shapes
:???????????20
.conv1d_transpose_2/conv1d_transpose/ExpandDims?
?conv1d_transpose_2/conv1d_transpose/ExpandDims_1/ReadVariableOpReadVariableOpHconv1d_transpose_2_conv1d_transpose_expanddims_1_readvariableop_resource*$
_output_shapes
:??*
dtype02A
?conv1d_transpose_2/conv1d_transpose/ExpandDims_1/ReadVariableOp?
4conv1d_transpose_2/conv1d_transpose/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 26
4conv1d_transpose_2/conv1d_transpose/ExpandDims_1/dim?
0conv1d_transpose_2/conv1d_transpose/ExpandDims_1
ExpandDimsGconv1d_transpose_2/conv1d_transpose/ExpandDims_1/ReadVariableOp:value:0=conv1d_transpose_2/conv1d_transpose/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:??22
0conv1d_transpose_2/conv1d_transpose/ExpandDims_1?
7conv1d_transpose_2/conv1d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 29
7conv1d_transpose_2/conv1d_transpose/strided_slice/stack?
9conv1d_transpose_2/conv1d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2;
9conv1d_transpose_2/conv1d_transpose/strided_slice/stack_1?
9conv1d_transpose_2/conv1d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2;
9conv1d_transpose_2/conv1d_transpose/strided_slice/stack_2?
1conv1d_transpose_2/conv1d_transpose/strided_sliceStridedSlice!conv1d_transpose_2/stack:output:0@conv1d_transpose_2/conv1d_transpose/strided_slice/stack:output:0Bconv1d_transpose_2/conv1d_transpose/strided_slice/stack_1:output:0Bconv1d_transpose_2/conv1d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask23
1conv1d_transpose_2/conv1d_transpose/strided_slice?
9conv1d_transpose_2/conv1d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2;
9conv1d_transpose_2/conv1d_transpose/strided_slice_1/stack?
;conv1d_transpose_2/conv1d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2=
;conv1d_transpose_2/conv1d_transpose/strided_slice_1/stack_1?
;conv1d_transpose_2/conv1d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2=
;conv1d_transpose_2/conv1d_transpose/strided_slice_1/stack_2?
3conv1d_transpose_2/conv1d_transpose/strided_slice_1StridedSlice!conv1d_transpose_2/stack:output:0Bconv1d_transpose_2/conv1d_transpose/strided_slice_1/stack:output:0Dconv1d_transpose_2/conv1d_transpose/strided_slice_1/stack_1:output:0Dconv1d_transpose_2/conv1d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask25
3conv1d_transpose_2/conv1d_transpose/strided_slice_1?
3conv1d_transpose_2/conv1d_transpose/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:25
3conv1d_transpose_2/conv1d_transpose/concat/values_1?
/conv1d_transpose_2/conv1d_transpose/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/conv1d_transpose_2/conv1d_transpose/concat/axis?
*conv1d_transpose_2/conv1d_transpose/concatConcatV2:conv1d_transpose_2/conv1d_transpose/strided_slice:output:0<conv1d_transpose_2/conv1d_transpose/concat/values_1:output:0<conv1d_transpose_2/conv1d_transpose/strided_slice_1:output:08conv1d_transpose_2/conv1d_transpose/concat/axis:output:0*
N*
T0*
_output_shapes
:2,
*conv1d_transpose_2/conv1d_transpose/concat?
#conv1d_transpose_2/conv1d_transposeConv2DBackpropInput3conv1d_transpose_2/conv1d_transpose/concat:output:09conv1d_transpose_2/conv1d_transpose/ExpandDims_1:output:07conv1d_transpose_2/conv1d_transpose/ExpandDims:output:0*
T0*9
_output_shapes'
%:#???????????????????*
paddingSAME*
strides
2%
#conv1d_transpose_2/conv1d_transpose?
+conv1d_transpose_2/conv1d_transpose/SqueezeSqueeze,conv1d_transpose_2/conv1d_transpose:output:0*
T0*-
_output_shapes
:???????????*
squeeze_dims
2-
+conv1d_transpose_2/conv1d_transpose/Squeeze?
leaky_re_lu_3/LeakyRelu	LeakyRelu4conv1d_transpose_2/conv1d_transpose/Squeeze:output:0*-
_output_shapes
:???????????*
alpha%???>2
leaky_re_lu_3/LeakyRelu?
conv1d_transpose_3/ShapeShape%leaky_re_lu_3/LeakyRelu:activations:0*
T0*
_output_shapes
:2
conv1d_transpose_3/Shape?
&conv1d_transpose_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv1d_transpose_3/strided_slice/stack?
(conv1d_transpose_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv1d_transpose_3/strided_slice/stack_1?
(conv1d_transpose_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv1d_transpose_3/strided_slice/stack_2?
 conv1d_transpose_3/strided_sliceStridedSlice!conv1d_transpose_3/Shape:output:0/conv1d_transpose_3/strided_slice/stack:output:01conv1d_transpose_3/strided_slice/stack_1:output:01conv1d_transpose_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv1d_transpose_3/strided_slice?
(conv1d_transpose_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2*
(conv1d_transpose_3/strided_slice_1/stack?
*conv1d_transpose_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv1d_transpose_3/strided_slice_1/stack_1?
*conv1d_transpose_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv1d_transpose_3/strided_slice_1/stack_2?
"conv1d_transpose_3/strided_slice_1StridedSlice!conv1d_transpose_3/Shape:output:01conv1d_transpose_3/strided_slice_1/stack:output:03conv1d_transpose_3/strided_slice_1/stack_1:output:03conv1d_transpose_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv1d_transpose_3/strided_slice_1v
conv1d_transpose_3/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
conv1d_transpose_3/mul/y?
conv1d_transpose_3/mulMul+conv1d_transpose_3/strided_slice_1:output:0!conv1d_transpose_3/mul/y:output:0*
T0*
_output_shapes
: 2
conv1d_transpose_3/mulz
conv1d_transpose_3/stack/2Const*
_output_shapes
: *
dtype0*
value	B :@2
conv1d_transpose_3/stack/2?
conv1d_transpose_3/stackPack)conv1d_transpose_3/strided_slice:output:0conv1d_transpose_3/mul:z:0#conv1d_transpose_3/stack/2:output:0*
N*
T0*
_output_shapes
:2
conv1d_transpose_3/stack?
2conv1d_transpose_3/conv1d_transpose/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :24
2conv1d_transpose_3/conv1d_transpose/ExpandDims/dim?
.conv1d_transpose_3/conv1d_transpose/ExpandDims
ExpandDims%leaky_re_lu_3/LeakyRelu:activations:0;conv1d_transpose_3/conv1d_transpose/ExpandDims/dim:output:0*
T0*1
_output_shapes
:???????????20
.conv1d_transpose_3/conv1d_transpose/ExpandDims?
?conv1d_transpose_3/conv1d_transpose/ExpandDims_1/ReadVariableOpReadVariableOpHconv1d_transpose_3_conv1d_transpose_expanddims_1_readvariableop_resource*#
_output_shapes
:@?*
dtype02A
?conv1d_transpose_3/conv1d_transpose/ExpandDims_1/ReadVariableOp?
4conv1d_transpose_3/conv1d_transpose/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 26
4conv1d_transpose_3/conv1d_transpose/ExpandDims_1/dim?
0conv1d_transpose_3/conv1d_transpose/ExpandDims_1
ExpandDimsGconv1d_transpose_3/conv1d_transpose/ExpandDims_1/ReadVariableOp:value:0=conv1d_transpose_3/conv1d_transpose/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:@?22
0conv1d_transpose_3/conv1d_transpose/ExpandDims_1?
7conv1d_transpose_3/conv1d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 29
7conv1d_transpose_3/conv1d_transpose/strided_slice/stack?
9conv1d_transpose_3/conv1d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2;
9conv1d_transpose_3/conv1d_transpose/strided_slice/stack_1?
9conv1d_transpose_3/conv1d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2;
9conv1d_transpose_3/conv1d_transpose/strided_slice/stack_2?
1conv1d_transpose_3/conv1d_transpose/strided_sliceStridedSlice!conv1d_transpose_3/stack:output:0@conv1d_transpose_3/conv1d_transpose/strided_slice/stack:output:0Bconv1d_transpose_3/conv1d_transpose/strided_slice/stack_1:output:0Bconv1d_transpose_3/conv1d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask23
1conv1d_transpose_3/conv1d_transpose/strided_slice?
9conv1d_transpose_3/conv1d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2;
9conv1d_transpose_3/conv1d_transpose/strided_slice_1/stack?
;conv1d_transpose_3/conv1d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2=
;conv1d_transpose_3/conv1d_transpose/strided_slice_1/stack_1?
;conv1d_transpose_3/conv1d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2=
;conv1d_transpose_3/conv1d_transpose/strided_slice_1/stack_2?
3conv1d_transpose_3/conv1d_transpose/strided_slice_1StridedSlice!conv1d_transpose_3/stack:output:0Bconv1d_transpose_3/conv1d_transpose/strided_slice_1/stack:output:0Dconv1d_transpose_3/conv1d_transpose/strided_slice_1/stack_1:output:0Dconv1d_transpose_3/conv1d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask25
3conv1d_transpose_3/conv1d_transpose/strided_slice_1?
3conv1d_transpose_3/conv1d_transpose/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:25
3conv1d_transpose_3/conv1d_transpose/concat/values_1?
/conv1d_transpose_3/conv1d_transpose/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/conv1d_transpose_3/conv1d_transpose/concat/axis?
*conv1d_transpose_3/conv1d_transpose/concatConcatV2:conv1d_transpose_3/conv1d_transpose/strided_slice:output:0<conv1d_transpose_3/conv1d_transpose/concat/values_1:output:0<conv1d_transpose_3/conv1d_transpose/strided_slice_1:output:08conv1d_transpose_3/conv1d_transpose/concat/axis:output:0*
N*
T0*
_output_shapes
:2,
*conv1d_transpose_3/conv1d_transpose/concat?
#conv1d_transpose_3/conv1d_transposeConv2DBackpropInput3conv1d_transpose_3/conv1d_transpose/concat:output:09conv1d_transpose_3/conv1d_transpose/ExpandDims_1:output:07conv1d_transpose_3/conv1d_transpose/ExpandDims:output:0*
T0*8
_output_shapes&
$:"??????????????????@*
paddingSAME*
strides
2%
#conv1d_transpose_3/conv1d_transpose?
+conv1d_transpose_3/conv1d_transpose/SqueezeSqueeze,conv1d_transpose_3/conv1d_transpose:output:0*
T0*,
_output_shapes
:?????????? @*
squeeze_dims
2-
+conv1d_transpose_3/conv1d_transpose/Squeeze?
leaky_re_lu_4/LeakyRelu	LeakyRelu4conv1d_transpose_3/conv1d_transpose/Squeeze:output:0*,
_output_shapes
:?????????? @*
alpha%???>2
leaky_re_lu_4/LeakyRelu?
conv1d_transpose_4/ShapeShape%leaky_re_lu_4/LeakyRelu:activations:0*
T0*
_output_shapes
:2
conv1d_transpose_4/Shape?
&conv1d_transpose_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv1d_transpose_4/strided_slice/stack?
(conv1d_transpose_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv1d_transpose_4/strided_slice/stack_1?
(conv1d_transpose_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv1d_transpose_4/strided_slice/stack_2?
 conv1d_transpose_4/strided_sliceStridedSlice!conv1d_transpose_4/Shape:output:0/conv1d_transpose_4/strided_slice/stack:output:01conv1d_transpose_4/strided_slice/stack_1:output:01conv1d_transpose_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv1d_transpose_4/strided_slice?
(conv1d_transpose_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2*
(conv1d_transpose_4/strided_slice_1/stack?
*conv1d_transpose_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv1d_transpose_4/strided_slice_1/stack_1?
*conv1d_transpose_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv1d_transpose_4/strided_slice_1/stack_2?
"conv1d_transpose_4/strided_slice_1StridedSlice!conv1d_transpose_4/Shape:output:01conv1d_transpose_4/strided_slice_1/stack:output:03conv1d_transpose_4/strided_slice_1/stack_1:output:03conv1d_transpose_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv1d_transpose_4/strided_slice_1v
conv1d_transpose_4/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
conv1d_transpose_4/mul/y?
conv1d_transpose_4/mulMul+conv1d_transpose_4/strided_slice_1:output:0!conv1d_transpose_4/mul/y:output:0*
T0*
_output_shapes
: 2
conv1d_transpose_4/mulz
conv1d_transpose_4/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
conv1d_transpose_4/stack/2?
conv1d_transpose_4/stackPack)conv1d_transpose_4/strided_slice:output:0conv1d_transpose_4/mul:z:0#conv1d_transpose_4/stack/2:output:0*
N*
T0*
_output_shapes
:2
conv1d_transpose_4/stack?
2conv1d_transpose_4/conv1d_transpose/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :24
2conv1d_transpose_4/conv1d_transpose/ExpandDims/dim?
.conv1d_transpose_4/conv1d_transpose/ExpandDims
ExpandDims%leaky_re_lu_4/LeakyRelu:activations:0;conv1d_transpose_4/conv1d_transpose/ExpandDims/dim:output:0*
T0*0
_output_shapes
:?????????? @20
.conv1d_transpose_4/conv1d_transpose/ExpandDims?
?conv1d_transpose_4/conv1d_transpose/ExpandDims_1/ReadVariableOpReadVariableOpHconv1d_transpose_4_conv1d_transpose_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02A
?conv1d_transpose_4/conv1d_transpose/ExpandDims_1/ReadVariableOp?
4conv1d_transpose_4/conv1d_transpose/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 26
4conv1d_transpose_4/conv1d_transpose/ExpandDims_1/dim?
0conv1d_transpose_4/conv1d_transpose/ExpandDims_1
ExpandDimsGconv1d_transpose_4/conv1d_transpose/ExpandDims_1/ReadVariableOp:value:0=conv1d_transpose_4/conv1d_transpose/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@22
0conv1d_transpose_4/conv1d_transpose/ExpandDims_1?
7conv1d_transpose_4/conv1d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 29
7conv1d_transpose_4/conv1d_transpose/strided_slice/stack?
9conv1d_transpose_4/conv1d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2;
9conv1d_transpose_4/conv1d_transpose/strided_slice/stack_1?
9conv1d_transpose_4/conv1d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2;
9conv1d_transpose_4/conv1d_transpose/strided_slice/stack_2?
1conv1d_transpose_4/conv1d_transpose/strided_sliceStridedSlice!conv1d_transpose_4/stack:output:0@conv1d_transpose_4/conv1d_transpose/strided_slice/stack:output:0Bconv1d_transpose_4/conv1d_transpose/strided_slice/stack_1:output:0Bconv1d_transpose_4/conv1d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask23
1conv1d_transpose_4/conv1d_transpose/strided_slice?
9conv1d_transpose_4/conv1d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2;
9conv1d_transpose_4/conv1d_transpose/strided_slice_1/stack?
;conv1d_transpose_4/conv1d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2=
;conv1d_transpose_4/conv1d_transpose/strided_slice_1/stack_1?
;conv1d_transpose_4/conv1d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2=
;conv1d_transpose_4/conv1d_transpose/strided_slice_1/stack_2?
3conv1d_transpose_4/conv1d_transpose/strided_slice_1StridedSlice!conv1d_transpose_4/stack:output:0Bconv1d_transpose_4/conv1d_transpose/strided_slice_1/stack:output:0Dconv1d_transpose_4/conv1d_transpose/strided_slice_1/stack_1:output:0Dconv1d_transpose_4/conv1d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask25
3conv1d_transpose_4/conv1d_transpose/strided_slice_1?
3conv1d_transpose_4/conv1d_transpose/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:25
3conv1d_transpose_4/conv1d_transpose/concat/values_1?
/conv1d_transpose_4/conv1d_transpose/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/conv1d_transpose_4/conv1d_transpose/concat/axis?
*conv1d_transpose_4/conv1d_transpose/concatConcatV2:conv1d_transpose_4/conv1d_transpose/strided_slice:output:0<conv1d_transpose_4/conv1d_transpose/concat/values_1:output:0<conv1d_transpose_4/conv1d_transpose/strided_slice_1:output:08conv1d_transpose_4/conv1d_transpose/concat/axis:output:0*
N*
T0*
_output_shapes
:2,
*conv1d_transpose_4/conv1d_transpose/concat?
#conv1d_transpose_4/conv1d_transposeConv2DBackpropInput3conv1d_transpose_4/conv1d_transpose/concat:output:09conv1d_transpose_4/conv1d_transpose/ExpandDims_1:output:07conv1d_transpose_4/conv1d_transpose/ExpandDims:output:0*
T0*8
_output_shapes&
$:"??????????????????*
paddingSAME*
strides
2%
#conv1d_transpose_4/conv1d_transpose?
+conv1d_transpose_4/conv1d_transpose/SqueezeSqueeze,conv1d_transpose_4/conv1d_transpose:output:0*
T0*-
_output_shapes
:???????????*
squeeze_dims
2-
+conv1d_transpose_4/conv1d_transpose/Squeeze?
conv1d_transpose_4/TanhTanh4conv1d_transpose_4/conv1d_transpose/Squeeze:output:0*
T0*-
_output_shapes
:???????????2
conv1d_transpose_4/Tanh|
IdentityIdentityconv1d_transpose_4/Tanh:y:0^NoOp*
T0*-
_output_shapes
:???????????2

Identity?
NoOpNoOp>^conv1d_transpose/conv1d_transpose/ExpandDims_1/ReadVariableOp@^conv1d_transpose_1/conv1d_transpose/ExpandDims_1/ReadVariableOp@^conv1d_transpose_2/conv1d_transpose/ExpandDims_1/ReadVariableOp@^conv1d_transpose_3/conv1d_transpose/ExpandDims_1/ReadVariableOp@^conv1d_transpose_4/conv1d_transpose/ExpandDims_1/ReadVariableOp^dense/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : : : : : 2~
=conv1d_transpose/conv1d_transpose/ExpandDims_1/ReadVariableOp=conv1d_transpose/conv1d_transpose/ExpandDims_1/ReadVariableOp2?
?conv1d_transpose_1/conv1d_transpose/ExpandDims_1/ReadVariableOp?conv1d_transpose_1/conv1d_transpose/ExpandDims_1/ReadVariableOp2?
?conv1d_transpose_2/conv1d_transpose/ExpandDims_1/ReadVariableOp?conv1d_transpose_2/conv1d_transpose/ExpandDims_1/ReadVariableOp2?
?conv1d_transpose_3/conv1d_transpose/ExpandDims_1/ReadVariableOp?conv1d_transpose_3/conv1d_transpose/ExpandDims_1/ReadVariableOp2?
?conv1d_transpose_4/conv1d_transpose/ExpandDims_1/ReadVariableOp?conv1d_transpose_4/conv1d_transpose/ExpandDims_1/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?.
?
N__inference_conv1d_transpose_layer_call_and_return_conditional_losses_78010833

inputsM
5conv1d_transpose_expanddims_1_readvariableop_resource:??
identity??,conv1d_transpose/ExpandDims_1/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulU
stack/2Const*
_output_shapes
: *
dtype0*
value
B :?2	
stack/2w
stackPackstrided_slice:output:0mul:z:0stack/2:output:0*
N*
T0*
_output_shapes
:2
stack?
conv1d_transpose/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2!
conv1d_transpose/ExpandDims/dim?
conv1d_transpose/ExpandDims
ExpandDimsinputs(conv1d_transpose/ExpandDims/dim:output:0*
T0*9
_output_shapes'
%:#???????????????????2
conv1d_transpose/ExpandDims?
,conv1d_transpose/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_transpose_expanddims_1_readvariableop_resource*$
_output_shapes
:??*
dtype02.
,conv1d_transpose/ExpandDims_1/ReadVariableOp?
!conv1d_transpose/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!conv1d_transpose/ExpandDims_1/dim?
conv1d_transpose/ExpandDims_1
ExpandDims4conv1d_transpose/ExpandDims_1/ReadVariableOp:value:0*conv1d_transpose/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:??2
conv1d_transpose/ExpandDims_1?
$conv1d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$conv1d_transpose/strided_slice/stack?
&conv1d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv1d_transpose/strided_slice/stack_1?
&conv1d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv1d_transpose/strided_slice/stack_2?
conv1d_transpose/strided_sliceStridedSlicestack:output:0-conv1d_transpose/strided_slice/stack:output:0/conv1d_transpose/strided_slice/stack_1:output:0/conv1d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2 
conv1d_transpose/strided_slice?
&conv1d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2(
&conv1d_transpose/strided_slice_1/stack?
(conv1d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(conv1d_transpose/strided_slice_1/stack_1?
(conv1d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv1d_transpose/strided_slice_1/stack_2?
 conv1d_transpose/strided_slice_1StridedSlicestack:output:0/conv1d_transpose/strided_slice_1/stack:output:01conv1d_transpose/strided_slice_1/stack_1:output:01conv1d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2"
 conv1d_transpose/strided_slice_1?
 conv1d_transpose/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 conv1d_transpose/concat/values_1~
conv1d_transpose/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d_transpose/concat/axis?
conv1d_transpose/concatConcatV2'conv1d_transpose/strided_slice:output:0)conv1d_transpose/concat/values_1:output:0)conv1d_transpose/strided_slice_1:output:0%conv1d_transpose/concat/axis:output:0*
N*
T0*
_output_shapes
:2
conv1d_transpose/concat?
conv1d_transposeConv2DBackpropInput conv1d_transpose/concat:output:0&conv1d_transpose/ExpandDims_1:output:0$conv1d_transpose/ExpandDims:output:0*
T0*9
_output_shapes'
%:#???????????????????*
paddingSAME*
strides
2
conv1d_transpose?
conv1d_transpose/SqueezeSqueezeconv1d_transpose:output:0*
T0*5
_output_shapes#
!:???????????????????*
squeeze_dims
2
conv1d_transpose/Squeeze?
IdentityIdentity!conv1d_transpose/Squeeze:output:0^NoOp*
T0*5
_output_shapes#
!:???????????????????2

Identity}
NoOpNoOp-^conv1d_transpose/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:???????????????????: 2\
,conv1d_transpose/ExpandDims_1/ReadVariableOp,conv1d_transpose/ExpandDims_1/ReadVariableOp:] Y
5
_output_shapes#
!:???????????????????
 
_user_specified_nameinputs
?
?
5__inference_conv1d_transpose_2_layer_call_fn_78012466

inputs
unknown:??
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:???????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_conv1d_transpose_2_layer_call_and_return_conditional_losses_780114022
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*-
_output_shapes
:???????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:???????????: 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
-
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
5__inference_conv1d_transpose_4_layer_call_fn_78012660

inputs
unknown:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:???????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_conv1d_transpose_4_layer_call_and_return_conditional_losses_780114952
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*-
_output_shapes
:???????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*-
_input_shapes
:?????????? @: 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:?????????? @
 
_user_specified_nameinputs
?6
?
J__inference_WG_Generator_layer_call_and_return_conditional_losses_78011500

inputs"
dense_78011250:
??1
conv1d_transpose_78011311:??3
conv1d_transpose_1_78011357:??3
conv1d_transpose_2_78011403:??2
conv1d_transpose_3_78011449:@?1
conv1d_transpose_4_78011496:@
identity??(conv1d_transpose/StatefulPartitionedCall?*conv1d_transpose_1/StatefulPartitionedCall?*conv1d_transpose_2/StatefulPartitionedCall?*conv1d_transpose_3/StatefulPartitionedCall?*conv1d_transpose_4/StatefulPartitionedCall?dense/StatefulPartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCallinputsdense_78011250*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_layer_call_and_return_conditional_losses_780112492
dense/StatefulPartitionedCall?
leaky_re_lu/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_leaky_re_lu_layer_call_and_return_conditional_losses_780112582
leaky_re_lu/PartitionedCall?
reshape/PartitionedCallPartitionedCall$leaky_re_lu/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_reshape_layer_call_and_return_conditional_losses_780112732
reshape/PartitionedCall?
(conv1d_transpose/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0conv1d_transpose_78011311*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????@?*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_conv1d_transpose_layer_call_and_return_conditional_losses_780113102*
(conv1d_transpose/StatefulPartitionedCall?
leaky_re_lu_1/PartitionedCallPartitionedCall1conv1d_transpose/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????@?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_780113192
leaky_re_lu_1/PartitionedCall?
*conv1d_transpose_1/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_1/PartitionedCall:output:0conv1d_transpose_1_78011357*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:???????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_conv1d_transpose_1_layer_call_and_return_conditional_losses_780113562,
*conv1d_transpose_1/StatefulPartitionedCall?
leaky_re_lu_2/PartitionedCallPartitionedCall3conv1d_transpose_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_780113652
leaky_re_lu_2/PartitionedCall?
*conv1d_transpose_2/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_2/PartitionedCall:output:0conv1d_transpose_2_78011403*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:???????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_conv1d_transpose_2_layer_call_and_return_conditional_losses_780114022,
*conv1d_transpose_2/StatefulPartitionedCall?
leaky_re_lu_3/PartitionedCallPartitionedCall3conv1d_transpose_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_leaky_re_lu_3_layer_call_and_return_conditional_losses_780114112
leaky_re_lu_3/PartitionedCall?
*conv1d_transpose_3/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_3/PartitionedCall:output:0conv1d_transpose_3_78011449*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????? @*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_conv1d_transpose_3_layer_call_and_return_conditional_losses_780114482,
*conv1d_transpose_3/StatefulPartitionedCall?
leaky_re_lu_4/PartitionedCallPartitionedCall3conv1d_transpose_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????? @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_leaky_re_lu_4_layer_call_and_return_conditional_losses_780114572
leaky_re_lu_4/PartitionedCall?
*conv1d_transpose_4/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_4/PartitionedCall:output:0conv1d_transpose_4_78011496*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:???????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_conv1d_transpose_4_layer_call_and_return_conditional_losses_780114952,
*conv1d_transpose_4/StatefulPartitionedCall?
IdentityIdentity3conv1d_transpose_4/StatefulPartitionedCall:output:0^NoOp*
T0*-
_output_shapes
:???????????2

Identity?
NoOpNoOp)^conv1d_transpose/StatefulPartitionedCall+^conv1d_transpose_1/StatefulPartitionedCall+^conv1d_transpose_2/StatefulPartitionedCall+^conv1d_transpose_3/StatefulPartitionedCall+^conv1d_transpose_4/StatefulPartitionedCall^dense/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : : : : : 2T
(conv1d_transpose/StatefulPartitionedCall(conv1d_transpose/StatefulPartitionedCall2X
*conv1d_transpose_1/StatefulPartitionedCall*conv1d_transpose_1/StatefulPartitionedCall2X
*conv1d_transpose_2/StatefulPartitionedCall*conv1d_transpose_2/StatefulPartitionedCall2X
*conv1d_transpose_3/StatefulPartitionedCall*conv1d_transpose_3/StatefulPartitionedCall2X
*conv1d_transpose_4/StatefulPartitionedCall*conv1d_transpose_4/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
/__inference_WG_Generator_layer_call_fn_78011673
input_1
unknown:
??!
	unknown_0:??!
	unknown_1:??!
	unknown_2:?? 
	unknown_3:@?
	unknown_4:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:???????????*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_WG_Generator_layer_call_and_return_conditional_losses_780116412
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*-
_output_shapes
:???????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1
?.
?
P__inference_conv1d_transpose_3_layer_call_and_return_conditional_losses_78011448

inputsL
5conv1d_transpose_expanddims_1_readvariableop_resource:@?
identity??,conv1d_transpose/ExpandDims_1/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulT
stack/2Const*
_output_shapes
: *
dtype0*
value	B :@2	
stack/2w
stackPackstrided_slice:output:0mul:z:0stack/2:output:0*
N*
T0*
_output_shapes
:2
stack?
conv1d_transpose/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2!
conv1d_transpose/ExpandDims/dim?
conv1d_transpose/ExpandDims
ExpandDimsinputs(conv1d_transpose/ExpandDims/dim:output:0*
T0*1
_output_shapes
:???????????2
conv1d_transpose/ExpandDims?
,conv1d_transpose/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_transpose_expanddims_1_readvariableop_resource*#
_output_shapes
:@?*
dtype02.
,conv1d_transpose/ExpandDims_1/ReadVariableOp?
!conv1d_transpose/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!conv1d_transpose/ExpandDims_1/dim?
conv1d_transpose/ExpandDims_1
ExpandDims4conv1d_transpose/ExpandDims_1/ReadVariableOp:value:0*conv1d_transpose/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:@?2
conv1d_transpose/ExpandDims_1?
$conv1d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$conv1d_transpose/strided_slice/stack?
&conv1d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv1d_transpose/strided_slice/stack_1?
&conv1d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv1d_transpose/strided_slice/stack_2?
conv1d_transpose/strided_sliceStridedSlicestack:output:0-conv1d_transpose/strided_slice/stack:output:0/conv1d_transpose/strided_slice/stack_1:output:0/conv1d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2 
conv1d_transpose/strided_slice?
&conv1d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2(
&conv1d_transpose/strided_slice_1/stack?
(conv1d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(conv1d_transpose/strided_slice_1/stack_1?
(conv1d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv1d_transpose/strided_slice_1/stack_2?
 conv1d_transpose/strided_slice_1StridedSlicestack:output:0/conv1d_transpose/strided_slice_1/stack:output:01conv1d_transpose/strided_slice_1/stack_1:output:01conv1d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2"
 conv1d_transpose/strided_slice_1?
 conv1d_transpose/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 conv1d_transpose/concat/values_1~
conv1d_transpose/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d_transpose/concat/axis?
conv1d_transpose/concatConcatV2'conv1d_transpose/strided_slice:output:0)conv1d_transpose/concat/values_1:output:0)conv1d_transpose/strided_slice_1:output:0%conv1d_transpose/concat/axis:output:0*
N*
T0*
_output_shapes
:2
conv1d_transpose/concat?
conv1d_transposeConv2DBackpropInput conv1d_transpose/concat:output:0&conv1d_transpose/ExpandDims_1:output:0$conv1d_transpose/ExpandDims:output:0*
T0*8
_output_shapes&
$:"??????????????????@*
paddingSAME*
strides
2
conv1d_transpose?
conv1d_transpose/SqueezeSqueezeconv1d_transpose:output:0*
T0*,
_output_shapes
:?????????? @*
squeeze_dims
2
conv1d_transpose/Squeeze?
IdentityIdentity!conv1d_transpose/Squeeze:output:0^NoOp*
T0*,
_output_shapes
:?????????? @2

Identity}
NoOpNoOp-^conv1d_transpose/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:???????????: 2\
,conv1d_transpose/ExpandDims_1/ReadVariableOp,conv1d_transpose/ExpandDims_1/ReadVariableOp:U Q
-
_output_shapes
:???????????
 
_user_specified_nameinputs
?	
?
&__inference_signature_wrapper_78011748
input_1
unknown:
??!
	unknown_0:??!
	unknown_1:??!
	unknown_2:?? 
	unknown_3:@?
	unknown_4:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:???????????*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *,
f'R%
#__inference__wrapped_model_780107932
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*-
_output_shapes
:???????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1
?
?
5__inference_conv1d_transpose_1_layer_call_fn_78012370

inputs
unknown:??
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:???????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_conv1d_transpose_1_layer_call_and_return_conditional_losses_780113562
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*-
_output_shapes
:???????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*-
_input_shapes
:?????????@?: 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:?????????@?
 
_user_specified_nameinputs
?
~
(__inference_dense_layer_call_fn_78012160

inputs
unknown:
??
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_layer_call_and_return_conditional_losses_780112492
StatefulPartitionedCall}
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*)
_output_shapes
:???????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?6
?
J__inference_WG_Generator_layer_call_and_return_conditional_losses_78011701
input_1"
dense_78011676:
??1
conv1d_transpose_78011681:??3
conv1d_transpose_1_78011685:??3
conv1d_transpose_2_78011689:??2
conv1d_transpose_3_78011693:@?1
conv1d_transpose_4_78011697:@
identity??(conv1d_transpose/StatefulPartitionedCall?*conv1d_transpose_1/StatefulPartitionedCall?*conv1d_transpose_2/StatefulPartitionedCall?*conv1d_transpose_3/StatefulPartitionedCall?*conv1d_transpose_4/StatefulPartitionedCall?dense/StatefulPartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCallinput_1dense_78011676*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_layer_call_and_return_conditional_losses_780112492
dense/StatefulPartitionedCall?
leaky_re_lu/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_leaky_re_lu_layer_call_and_return_conditional_losses_780112582
leaky_re_lu/PartitionedCall?
reshape/PartitionedCallPartitionedCall$leaky_re_lu/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_reshape_layer_call_and_return_conditional_losses_780112732
reshape/PartitionedCall?
(conv1d_transpose/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0conv1d_transpose_78011681*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????@?*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_conv1d_transpose_layer_call_and_return_conditional_losses_780113102*
(conv1d_transpose/StatefulPartitionedCall?
leaky_re_lu_1/PartitionedCallPartitionedCall1conv1d_transpose/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????@?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_780113192
leaky_re_lu_1/PartitionedCall?
*conv1d_transpose_1/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_1/PartitionedCall:output:0conv1d_transpose_1_78011685*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:???????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_conv1d_transpose_1_layer_call_and_return_conditional_losses_780113562,
*conv1d_transpose_1/StatefulPartitionedCall?
leaky_re_lu_2/PartitionedCallPartitionedCall3conv1d_transpose_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_780113652
leaky_re_lu_2/PartitionedCall?
*conv1d_transpose_2/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_2/PartitionedCall:output:0conv1d_transpose_2_78011689*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:???????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_conv1d_transpose_2_layer_call_and_return_conditional_losses_780114022,
*conv1d_transpose_2/StatefulPartitionedCall?
leaky_re_lu_3/PartitionedCallPartitionedCall3conv1d_transpose_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_leaky_re_lu_3_layer_call_and_return_conditional_losses_780114112
leaky_re_lu_3/PartitionedCall?
*conv1d_transpose_3/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_3/PartitionedCall:output:0conv1d_transpose_3_78011693*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????? @*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_conv1d_transpose_3_layer_call_and_return_conditional_losses_780114482,
*conv1d_transpose_3/StatefulPartitionedCall?
leaky_re_lu_4/PartitionedCallPartitionedCall3conv1d_transpose_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????? @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_leaky_re_lu_4_layer_call_and_return_conditional_losses_780114572
leaky_re_lu_4/PartitionedCall?
*conv1d_transpose_4/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_4/PartitionedCall:output:0conv1d_transpose_4_78011697*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:???????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_conv1d_transpose_4_layer_call_and_return_conditional_losses_780114952,
*conv1d_transpose_4/StatefulPartitionedCall?
IdentityIdentity3conv1d_transpose_4/StatefulPartitionedCall:output:0^NoOp*
T0*-
_output_shapes
:???????????2

Identity?
NoOpNoOp)^conv1d_transpose/StatefulPartitionedCall+^conv1d_transpose_1/StatefulPartitionedCall+^conv1d_transpose_2/StatefulPartitionedCall+^conv1d_transpose_3/StatefulPartitionedCall+^conv1d_transpose_4/StatefulPartitionedCall^dense/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : : : : : 2T
(conv1d_transpose/StatefulPartitionedCall(conv1d_transpose/StatefulPartitionedCall2X
*conv1d_transpose_1/StatefulPartitionedCall*conv1d_transpose_1/StatefulPartitionedCall2X
*conv1d_transpose_2/StatefulPartitionedCall*conv1d_transpose_2/StatefulPartitionedCall2X
*conv1d_transpose_3/StatefulPartitionedCall*conv1d_transpose_3/StatefulPartitionedCall2X
*conv1d_transpose_4/StatefulPartitionedCall*conv1d_transpose_4/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1
?/
?
P__inference_conv1d_transpose_4_layer_call_and_return_conditional_losses_78011495

inputsK
5conv1d_transpose_expanddims_1_readvariableop_resource:@
identity??,conv1d_transpose/ExpandDims_1/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulT
stack/2Const*
_output_shapes
: *
dtype0*
value	B :2	
stack/2w
stackPackstrided_slice:output:0mul:z:0stack/2:output:0*
N*
T0*
_output_shapes
:2
stack?
conv1d_transpose/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2!
conv1d_transpose/ExpandDims/dim?
conv1d_transpose/ExpandDims
ExpandDimsinputs(conv1d_transpose/ExpandDims/dim:output:0*
T0*0
_output_shapes
:?????????? @2
conv1d_transpose/ExpandDims?
,conv1d_transpose/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_transpose_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02.
,conv1d_transpose/ExpandDims_1/ReadVariableOp?
!conv1d_transpose/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!conv1d_transpose/ExpandDims_1/dim?
conv1d_transpose/ExpandDims_1
ExpandDims4conv1d_transpose/ExpandDims_1/ReadVariableOp:value:0*conv1d_transpose/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@2
conv1d_transpose/ExpandDims_1?
$conv1d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$conv1d_transpose/strided_slice/stack?
&conv1d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv1d_transpose/strided_slice/stack_1?
&conv1d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv1d_transpose/strided_slice/stack_2?
conv1d_transpose/strided_sliceStridedSlicestack:output:0-conv1d_transpose/strided_slice/stack:output:0/conv1d_transpose/strided_slice/stack_1:output:0/conv1d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2 
conv1d_transpose/strided_slice?
&conv1d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2(
&conv1d_transpose/strided_slice_1/stack?
(conv1d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(conv1d_transpose/strided_slice_1/stack_1?
(conv1d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv1d_transpose/strided_slice_1/stack_2?
 conv1d_transpose/strided_slice_1StridedSlicestack:output:0/conv1d_transpose/strided_slice_1/stack:output:01conv1d_transpose/strided_slice_1/stack_1:output:01conv1d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2"
 conv1d_transpose/strided_slice_1?
 conv1d_transpose/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 conv1d_transpose/concat/values_1~
conv1d_transpose/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d_transpose/concat/axis?
conv1d_transpose/concatConcatV2'conv1d_transpose/strided_slice:output:0)conv1d_transpose/concat/values_1:output:0)conv1d_transpose/strided_slice_1:output:0%conv1d_transpose/concat/axis:output:0*
N*
T0*
_output_shapes
:2
conv1d_transpose/concat?
conv1d_transposeConv2DBackpropInput conv1d_transpose/concat:output:0&conv1d_transpose/ExpandDims_1:output:0$conv1d_transpose/ExpandDims:output:0*
T0*8
_output_shapes&
$:"??????????????????*
paddingSAME*
strides
2
conv1d_transpose?
conv1d_transpose/SqueezeSqueezeconv1d_transpose:output:0*
T0*-
_output_shapes
:???????????*
squeeze_dims
2
conv1d_transpose/Squeezeo
TanhTanh!conv1d_transpose/Squeeze:output:0*
T0*-
_output_shapes
:???????????2
Tanhi
IdentityIdentityTanh:y:0^NoOp*
T0*-
_output_shapes
:???????????2

Identity}
NoOpNoOp-^conv1d_transpose/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*-
_input_shapes
:?????????? @: 2\
,conv1d_transpose/ExpandDims_1/ReadVariableOp,conv1d_transpose/ExpandDims_1/ReadVariableOp:T P
,
_output_shapes
:?????????? @
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
;
input_10
serving_default_input_1:0?????????L
conv1d_transpose_46
StatefulPartitionedCall:0???????????tensorflow/serving/predict:??
?
layer-0
layer_with_weights-0
layer-1
layer-2
layer-3
layer_with_weights-1
layer-4
layer-5
layer_with_weights-2
layer-6
layer-7
	layer_with_weights-3
	layer-8

layer-9
layer_with_weights-4
layer-10
layer-11
layer_with_weights-5
layer-12
regularization_losses
	variables
trainable_variables
	keras_api

signatures
+?&call_and_return_all_conditional_losses
?_default_save_signature
?__call__"
_tf_keras_network
"
_tf_keras_input_layer
?

kernel
regularization_losses
	variables
trainable_variables
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
regularization_losses
	variables
trainable_variables
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
regularization_losses
	variables
trainable_variables
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?

 kernel
!regularization_losses
"	variables
#trainable_variables
$	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
%regularization_losses
&	variables
'trainable_variables
(	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?

)kernel
*regularization_losses
+	variables
,trainable_variables
-	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
.regularization_losses
/	variables
0trainable_variables
1	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?

2kernel
3regularization_losses
4	variables
5trainable_variables
6	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
7regularization_losses
8	variables
9trainable_variables
:	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?

;kernel
<regularization_losses
=	variables
>trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
@regularization_losses
A	variables
Btrainable_variables
C	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?

Dkernel
Eregularization_losses
F	variables
Gtrainable_variables
H	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
 "
trackable_list_wrapper
J
0
 1
)2
23
;4
D5"
trackable_list_wrapper
J
0
 1
)2
23
;4
D5"
trackable_list_wrapper
?
Inon_trainable_variables

Jlayers
regularization_losses
Klayer_regularization_losses
Lmetrics
	variables
trainable_variables
Mlayer_metrics
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
 :
??2dense/kernel
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
?
Nnon_trainable_variables

Olayers
regularization_losses
Player_regularization_losses
Qmetrics
	variables
trainable_variables
Rlayer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Snon_trainable_variables

Tlayers
regularization_losses
Ulayer_regularization_losses
Vmetrics
	variables
trainable_variables
Wlayer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Xnon_trainable_variables

Ylayers
regularization_losses
Zlayer_regularization_losses
[metrics
	variables
trainable_variables
\layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
/:-??2conv1d_transpose/kernel
 "
trackable_list_wrapper
'
 0"
trackable_list_wrapper
'
 0"
trackable_list_wrapper
?
]non_trainable_variables

^layers
!regularization_losses
_layer_regularization_losses
`metrics
"	variables
#trainable_variables
alayer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
bnon_trainable_variables

clayers
%regularization_losses
dlayer_regularization_losses
emetrics
&	variables
'trainable_variables
flayer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
1:/??2conv1d_transpose_1/kernel
 "
trackable_list_wrapper
'
)0"
trackable_list_wrapper
'
)0"
trackable_list_wrapper
?
gnon_trainable_variables

hlayers
*regularization_losses
ilayer_regularization_losses
jmetrics
+	variables
,trainable_variables
klayer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
lnon_trainable_variables

mlayers
.regularization_losses
nlayer_regularization_losses
ometrics
/	variables
0trainable_variables
player_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
1:/??2conv1d_transpose_2/kernel
 "
trackable_list_wrapper
'
20"
trackable_list_wrapper
'
20"
trackable_list_wrapper
?
qnon_trainable_variables

rlayers
3regularization_losses
slayer_regularization_losses
tmetrics
4	variables
5trainable_variables
ulayer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
vnon_trainable_variables

wlayers
7regularization_losses
xlayer_regularization_losses
ymetrics
8	variables
9trainable_variables
zlayer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
0:.@?2conv1d_transpose_3/kernel
 "
trackable_list_wrapper
'
;0"
trackable_list_wrapper
'
;0"
trackable_list_wrapper
?
{non_trainable_variables

|layers
<regularization_losses
}layer_regularization_losses
~metrics
=	variables
>trainable_variables
layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
@regularization_losses
 ?layer_regularization_losses
?metrics
A	variables
Btrainable_variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
/:-@2conv1d_transpose_4/kernel
 "
trackable_list_wrapper
'
D0"
trackable_list_wrapper
'
D0"
trackable_list_wrapper
?
?non_trainable_variables
?layers
Eregularization_losses
 ?layer_regularization_losses
?metrics
F	variables
Gtrainable_variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
~
0
1
2
3
4
5
6
7
	8

9
10
11
12"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?2?
J__inference_WG_Generator_layer_call_and_return_conditional_losses_78011930
J__inference_WG_Generator_layer_call_and_return_conditional_losses_78012112
J__inference_WG_Generator_layer_call_and_return_conditional_losses_78011701
J__inference_WG_Generator_layer_call_and_return_conditional_losses_78011729?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
#__inference__wrapped_model_78010793input_1"?
???
FullArgSpec
args? 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
/__inference_WG_Generator_layer_call_fn_78011515
/__inference_WG_Generator_layer_call_fn_78012129
/__inference_WG_Generator_layer_call_fn_78012146
/__inference_WG_Generator_layer_call_fn_78011673?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
C__inference_dense_layer_call_and_return_conditional_losses_78012153?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_dense_layer_call_fn_78012160?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
I__inference_leaky_re_lu_layer_call_and_return_conditional_losses_78012165?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
.__inference_leaky_re_lu_layer_call_fn_78012170?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_reshape_layer_call_and_return_conditional_losses_78012183?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_reshape_layer_call_fn_78012188?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
N__inference_conv1d_transpose_layer_call_and_return_conditional_losses_78012224
N__inference_conv1d_transpose_layer_call_and_return_conditional_losses_78012260?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
3__inference_conv1d_transpose_layer_call_fn_78012267
3__inference_conv1d_transpose_layer_call_fn_78012274?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
K__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_78012279?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
0__inference_leaky_re_lu_1_layer_call_fn_78012284?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
P__inference_conv1d_transpose_1_layer_call_and_return_conditional_losses_78012320
P__inference_conv1d_transpose_1_layer_call_and_return_conditional_losses_78012356?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
5__inference_conv1d_transpose_1_layer_call_fn_78012363
5__inference_conv1d_transpose_1_layer_call_fn_78012370?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
K__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_78012375?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
0__inference_leaky_re_lu_2_layer_call_fn_78012380?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
P__inference_conv1d_transpose_2_layer_call_and_return_conditional_losses_78012416
P__inference_conv1d_transpose_2_layer_call_and_return_conditional_losses_78012452?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
5__inference_conv1d_transpose_2_layer_call_fn_78012459
5__inference_conv1d_transpose_2_layer_call_fn_78012466?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
K__inference_leaky_re_lu_3_layer_call_and_return_conditional_losses_78012471?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
0__inference_leaky_re_lu_3_layer_call_fn_78012476?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
P__inference_conv1d_transpose_3_layer_call_and_return_conditional_losses_78012512
P__inference_conv1d_transpose_3_layer_call_and_return_conditional_losses_78012548?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
5__inference_conv1d_transpose_3_layer_call_fn_78012555
5__inference_conv1d_transpose_3_layer_call_fn_78012562?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
K__inference_leaky_re_lu_4_layer_call_and_return_conditional_losses_78012567?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
0__inference_leaky_re_lu_4_layer_call_fn_78012572?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
P__inference_conv1d_transpose_4_layer_call_and_return_conditional_losses_78012609
P__inference_conv1d_transpose_4_layer_call_and_return_conditional_losses_78012646?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
5__inference_conv1d_transpose_4_layer_call_fn_78012653
5__inference_conv1d_transpose_4_layer_call_fn_78012660?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
&__inference_signature_wrapper_78011748input_1"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
J__inference_WG_Generator_layer_call_and_return_conditional_losses_78011701o )2;D8?5
.?+
!?
input_1?????????
p 

 
? "+?(
!?
0???????????
? ?
J__inference_WG_Generator_layer_call_and_return_conditional_losses_78011729o )2;D8?5
.?+
!?
input_1?????????
p

 
? "+?(
!?
0???????????
? ?
J__inference_WG_Generator_layer_call_and_return_conditional_losses_78011930n )2;D7?4
-?*
 ?
inputs?????????
p 

 
? "+?(
!?
0???????????
? ?
J__inference_WG_Generator_layer_call_and_return_conditional_losses_78012112n )2;D7?4
-?*
 ?
inputs?????????
p

 
? "+?(
!?
0???????????
? ?
/__inference_WG_Generator_layer_call_fn_78011515b )2;D8?5
.?+
!?
input_1?????????
p 

 
? "?????????????
/__inference_WG_Generator_layer_call_fn_78011673b )2;D8?5
.?+
!?
input_1?????????
p

 
? "?????????????
/__inference_WG_Generator_layer_call_fn_78012129a )2;D7?4
-?*
 ?
inputs?????????
p 

 
? "?????????????
/__inference_WG_Generator_layer_call_fn_78012146a )2;D7?4
-?*
 ?
inputs?????????
p

 
? "?????????????
#__inference__wrapped_model_78010793? )2;D0?-
&?#
!?
input_1?????????
? "M?J
H
conv1d_transpose_42?/
conv1d_transpose_4????????????
P__inference_conv1d_transpose_1_layer_call_and_return_conditional_losses_78012320w)=?:
3?0
.?+
inputs???????????????????
? "3?0
)?&
0???????????????????
? ?
P__inference_conv1d_transpose_1_layer_call_and_return_conditional_losses_78012356f)4?1
*?'
%?"
inputs?????????@?
? "+?(
!?
0???????????
? ?
5__inference_conv1d_transpose_1_layer_call_fn_78012363j)=?:
3?0
.?+
inputs???????????????????
? "&?#????????????????????
5__inference_conv1d_transpose_1_layer_call_fn_78012370Y)4?1
*?'
%?"
inputs?????????@?
? "?????????????
P__inference_conv1d_transpose_2_layer_call_and_return_conditional_losses_78012416w2=?:
3?0
.?+
inputs???????????????????
? "3?0
)?&
0???????????????????
? ?
P__inference_conv1d_transpose_2_layer_call_and_return_conditional_losses_78012452g25?2
+?(
&?#
inputs???????????
? "+?(
!?
0???????????
? ?
5__inference_conv1d_transpose_2_layer_call_fn_78012459j2=?:
3?0
.?+
inputs???????????????????
? "&?#????????????????????
5__inference_conv1d_transpose_2_layer_call_fn_78012466Z25?2
+?(
&?#
inputs???????????
? "?????????????
P__inference_conv1d_transpose_3_layer_call_and_return_conditional_losses_78012512v;=?:
3?0
.?+
inputs???????????????????
? "2?/
(?%
0??????????????????@
? ?
P__inference_conv1d_transpose_3_layer_call_and_return_conditional_losses_78012548f;5?2
+?(
&?#
inputs???????????
? "*?'
 ?
0?????????? @
? ?
5__inference_conv1d_transpose_3_layer_call_fn_78012555i;=?:
3?0
.?+
inputs???????????????????
? "%?"??????????????????@?
5__inference_conv1d_transpose_3_layer_call_fn_78012562Y;5?2
+?(
&?#
inputs???????????
? "??????????? @?
P__inference_conv1d_transpose_4_layer_call_and_return_conditional_losses_78012609uD<?9
2?/
-?*
inputs??????????????????@
? "2?/
(?%
0??????????????????
? ?
P__inference_conv1d_transpose_4_layer_call_and_return_conditional_losses_78012646fD4?1
*?'
%?"
inputs?????????? @
? "+?(
!?
0???????????
? ?
5__inference_conv1d_transpose_4_layer_call_fn_78012653hD<?9
2?/
-?*
inputs??????????????????@
? "%?"???????????????????
5__inference_conv1d_transpose_4_layer_call_fn_78012660YD4?1
*?'
%?"
inputs?????????? @
? "?????????????
N__inference_conv1d_transpose_layer_call_and_return_conditional_losses_78012224w =?:
3?0
.?+
inputs???????????????????
? "3?0
)?&
0???????????????????
? ?
N__inference_conv1d_transpose_layer_call_and_return_conditional_losses_78012260e 4?1
*?'
%?"
inputs??????????
? "*?'
 ?
0?????????@?
? ?
3__inference_conv1d_transpose_layer_call_fn_78012267j =?:
3?0
.?+
inputs???????????????????
? "&?#????????????????????
3__inference_conv1d_transpose_layer_call_fn_78012274X 4?1
*?'
%?"
inputs??????????
? "??????????@??
C__inference_dense_layer_call_and_return_conditional_losses_78012153]/?,
%?"
 ?
inputs?????????
? "'?$
?
0???????????
? |
(__inference_dense_layer_call_fn_78012160P/?,
%?"
 ?
inputs?????????
? "?????????????
K__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_78012279b4?1
*?'
%?"
inputs?????????@?
? "*?'
 ?
0?????????@?
? ?
0__inference_leaky_re_lu_1_layer_call_fn_78012284U4?1
*?'
%?"
inputs?????????@?
? "??????????@??
K__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_78012375d5?2
+?(
&?#
inputs???????????
? "+?(
!?
0???????????
? ?
0__inference_leaky_re_lu_2_layer_call_fn_78012380W5?2
+?(
&?#
inputs???????????
? "?????????????
K__inference_leaky_re_lu_3_layer_call_and_return_conditional_losses_78012471d5?2
+?(
&?#
inputs???????????
? "+?(
!?
0???????????
? ?
0__inference_leaky_re_lu_3_layer_call_fn_78012476W5?2
+?(
&?#
inputs???????????
? "?????????????
K__inference_leaky_re_lu_4_layer_call_and_return_conditional_losses_78012567b4?1
*?'
%?"
inputs?????????? @
? "*?'
 ?
0?????????? @
? ?
0__inference_leaky_re_lu_4_layer_call_fn_78012572U4?1
*?'
%?"
inputs?????????? @
? "??????????? @?
I__inference_leaky_re_lu_layer_call_and_return_conditional_losses_78012165\1?.
'?$
"?
inputs???????????
? "'?$
?
0???????????
? ?
.__inference_leaky_re_lu_layer_call_fn_78012170O1?.
'?$
"?
inputs???????????
? "?????????????
E__inference_reshape_layer_call_and_return_conditional_losses_78012183_1?.
'?$
"?
inputs???????????
? "*?'
 ?
0??????????
? ?
*__inference_reshape_layer_call_fn_78012188R1?.
'?$
"?
inputs???????????
? "????????????
&__inference_signature_wrapper_78011748? )2;D;?8
? 
1?.
,
input_1!?
input_1?????????"M?J
H
conv1d_transpose_42?/
conv1d_transpose_4???????????