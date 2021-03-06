?	??T??E@??T??E@!??T??E@	vT?'N??vT?'N??!vT?'N??"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6??T??E@^?zk`k@1o)狽]B@Ax'??2??I??}??k@YZ???Z???*	S??3+?@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::GeneratorX???D@!?o?	?X@)X???D@1?o?	?X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch??Ry=??!????W??)??Ry=??1????W??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism?l??)???!9?p????)a?unڌ??1??|IR???:Preprocessing2F
Iterator::Model.rOWw,??!
G??׮??)??q6??1B??Bi???:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap????D@!ݨ?(?X@)?????}?1Jď?E???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 9.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?5.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9uT?'N??I8]?ܭ.@QX?k?U@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	^?zk`k@^?zk`k@!^?zk`k@      ??!       "	o)狽]B@o)狽]B@!o)狽]B@*      ??!       2	x'??2??x'??2??!x'??2??:	??}??k@??}??k@!??}??k@B      ??!       J	Z???Z???Z???Z???!Z???Z???R      ??!       Z	Z???Z???Z???Z???!Z???Z???b      ??!       JGPUYuT?'N??b q8]?ܭ.@yX?k?U@?"j
>gradient_tape/sequential/conv2d_10/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter??^Q??!??^Q??0"h
=gradient_tape/sequential/conv2d_10/Conv2D/Conv2DBackpropInputConv2DBackpropInputO(??Ȱ?!^ 3\?{??0";
sequential/conv2d_10/Relu_FusedConv2D%wo?2ݮ?!o?~|???"i
=gradient_tape/sequential/conv2d_9/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter????!??????0"i
=gradient_tape/sequential/conv2d_8/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?Ei??E??!=??u???0"L
%Adam/Adam/update_20/ResourceApplyAdamResourceApplyAdam??Y$B??!2?ݝ%??"g
<gradient_tape/sequential/conv2d_9/Conv2D/Conv2DBackpropInputConv2DBackpropInput?̡?????!?G%u]??0"i
=gradient_tape/sequential/conv2d_2/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter???d???!?ڷ??}??0"i
=gradient_tape/sequential/conv2d_1/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter??}???!??tg???0":
sequential/conv2d_9/Relu_FusedConv2D䶼???!2W?e????Q      Y@Y     ?"@a     ?V@q??Q4y??y?ۇ?߄?"?

both?Your program is POTENTIALLY input-bound because 9.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
moderate?5.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQ2"GPU(: B 