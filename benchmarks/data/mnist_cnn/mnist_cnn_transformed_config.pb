ΐ(08@P`ΚκhζΕ ΈΠΐ1ΠΪπψ# |  [ 6 s  n M ( e  @ ? x  W 2  j  I $ a  \ ; t  S N f -  E   }  X p 7  o J ) b  A < y T  3 . k  F ~ %   ] 8 u  P O * g 	 B ! z  Y 4 q  l  K & c  ^ v =  U 0 / h 
 G "   Z 9 r  Q L d +  C > {  V 5  m  H ' `  _ : w  R 1 , i  D 5
M
!

J




b 
 
 ("reshape_transform.ReshapeTransform
Z
!

J




b 
 
 5/tf_matmul_transform.TFSavedModelMatMulTransform
Z
!

J




b 
 
 5/tf_conv2d_transform.TFSavedModelConv2DTransform
V
!

J




b 
 
 1+tf_pool_transform.TFSavedModelPoolTransform
j
!

J




b 
 
 E?tf_batch_norm_transform.TFSavedModelDecomposeBatchNormTransform
_
!

J




b 
 
 :4tf_pad_transform.TFSavedModelUnsupportedPadTransform
V
!

J
	



b 
 
 1	+tf_mean_transform.TFSavedModelMeanTransform
\
!

J





b 
 
 7
1tf_softmax_transform.TFSavedModelSoftmaxTransform
F
!

J




b 
 
 !vv_transform.VVAddTransform
E
!

J




b 
 
  sv_transform.ReluTransform
O
!

J




b 
 
 *$identity_transform.IdentityTransform
I
!

J




b 
 
 $relu6_transform.Relu6Transform
c
!

J




b 
 
 >8tf_conv2d_transform.TFSavedModelDepthwiseConv2DTransform
\
!

J




b 
 
 71tf_squeeze_transform.TFSavedModelSqueezeTransform
I
!

J




b 
 
 $swish_transform.SwishTransform
F
!

J




b 
 
 !vv_transform.VVMulTransform
M
!

J




b 
 
 ("sigmoid_transform.SigmoidTransform
G
!

J
 



b 
 
 " tanh_transform.TanhTransform
M
!

J
$



b 
 
 ($"reshape_transform.ReshapeTransform
V
!

J
%



b 
 
 1%+tf_fill_transform.TFSavedModelFillTransform"(0:U
matmul
conv2d
block_diagonal_depthwise_conv2d
distributed_depthwise_conv2dpool@ R p €§Ϊͺ° Έ ΐΪ

b 
 
 ς"*
CONV2D
VV_ADD
SV_MAX*
VV_ADD
VV_MUL
SV_MAX*6
DISTRIBUTED_DEPTHWISE_CONV2D
VV_ADD
SV_MAX
SV_MIN*
MATMUL
VV_ADD
SV_MAXϊ	EXPϊ
CONV2DϊVVXϊ NOOPϊ
ALL_REDUCE_SUMϊLDWϊSVXϊ
BUNDLEϊMOVEϊMAXPOOLϊAPWϊ
MATMULϊ	DW_CONV2DϊMEAN
SVX
VVX
	DW_CONV2D
ALL_REDUCE_SUM

MAXPOOL

CONV2D
LDW

MATMUL
MOVE
MEAN

BUNDLE
EXP	
NOOP 
APWΣ
LDW
mem_type
	pack_type
address	
count
mem_type
	pack_type
address	
count
mem_type
	pack_type
address	
count
sparsity
sparse_rows

unused_pad6
APW
quant_scale

quant_bias

unused_pad}
MOVE
mem_type
	pack_type
address"	
count
mem_type
	pack_type
address"	
count

unused_padτ
CONV2D
mem_type
	pack_type
address	
count
mem_type
	pack_type
address	
count
wb_mode
output_tiles
output_tile_size
kernel_size	
width

height
pad

stride

unused_pad"χ
	DW_CONV2D
mem_type
	pack_type
address	
count
mem_type
	pack_type
address	
count
wb_mode
output_tiles
output_tile_size
kernel_size	
width

height
pad

stride

unused_pad"
BUNDLE
mem_type
	pack_type
address	
count
mem_type
	pack_type
address	
count
wb_mode
output_tiles
output_tile_size
kernel_size	
width

height
pad

stride
bundle_type
sub_operand0
sub_operand1Ώ
VV_ADD
mem_type
	pack_type
address	
count
mem_type
	pack_type
address	
count
mem_type
	pack_type
address	
count
op

unused_padΏ
VV_MUL
mem_type
	pack_type
address	
count
mem_type
	pack_type
address	
count
mem_type
	pack_type
address	
count
op

unused_padΏ
VV_DIV
mem_type
	pack_type
address	
count
mem_type
	pack_type
address	
count
mem_type
	pack_type
address	
count
op

unused_pad
SV_ADD
mem_type
	pack_type
address	
count
mem_type
	pack_type
address	
count
op

scalar

unused_pad7
SV_MUL
mem_type
	pack_type
address	
count
mem_type
	pack_type
address	
count
op

scalar

unused_pad7
SV_MIN
mem_type
	pack_type
address	
count
mem_type
	pack_type
address	
count
op

scalar

unused_pad7
SV_MAX
mem_type
	pack_type
address	
count
mem_type
	pack_type
address	
count
op

scalar

unused_pad7ΐ
MAXPOOL
mem_type
	pack_type
address	
count
mem_type
	pack_type
address	
count
kernel_size	
width

height
pad

stride

unused_pad6
EXP
mem_type
	pack_type
address	
count
mem_type
	pack_type
address	
count

unused_padKX
GENERIC_INSTR

opcode	
x_dim	
y_dim	
i_dim
dep_pc_0
dep_pc_1Λ
SVX
mem_type
	pack_type
address	
count
mem_type
	pack_type
address	
count
op

scalar

unused_pad7
mem_type
	pack_type
address	
count
mem_type
	pack_type
address	
count
op

scalar

unused_pad7
mem_type
	pack_type
address	
count
mem_type
	pack_type
address	
count
op

scalar

unused_pad7
mem_type
	pack_type
address	
count
mem_type
	pack_type
address	
count
op

scalar

unused_pad7
mem_type
	pack_type
address	
count
mem_type
	pack_type
address	
count
op

scalar

unused_pad7α
VVX
mem_type
	pack_type
address	
count
mem_type
	pack_type
address	
count
mem_type
	pack_type
address	
count
op

unused_pad
mem_type
	pack_type
address	
count
mem_type
	pack_type
address	
count
mem_type
	pack_type
address	
count
op

unused_pad
mem_type
	pack_type
address	
count
mem_type
	pack_type
address	
count
mem_type
	pack_type
address	
count
op

unused_pad
mem_type
	pack_type
address	
count
mem_type
	pack_type
address	
count
mem_type
	pack_type
address	
count
op

unused_pad
MAX
EXPO
DIV
MIN
ADD 
MEAN	
ARS
BUNDLE_ADD_MAX_MIN
BUNDLE_MUL_ADD
MUL’-tf_const_transform.TFSavedModelConstTransform"ψ2H  zD  zD%(knQ-   B5(kξP=  zD@U   BX`hx ¨°ΈΠΨΰ:T  ΐ@  ΐ@  ΰ@%  °@-  ?5  ?=   @E  0AM  0AU  ΐ?]ffζ?e   @m  ?u   B}   A   @ @Z   @-   @5  ?M  ?}·Ρ5   A²1lt_sdk/simulation/opu_model/lut/SCIM_veriloga.csvΊ2lt_sdk/simulation/opu_model/lut/CIMZI_veriloga.csvΥ @Fε¨{7` hz9
  zD0






 (
08