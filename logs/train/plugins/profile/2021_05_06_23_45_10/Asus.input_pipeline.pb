	??D?m$?@??D?m$?@!??D?m$?@	pQ?B??`?pQ?B??`?!pQ?B??`?"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$??D?m$?@E??Ӝ???A??(y$?@Y?Hm????*	}?5^?`@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[6]::Concatenate???????!Wی???B@)?Ǻ????1?<??A@:Preprocessing2U
Iterator::Model::ParallelMapV2r??9???!wɌ\8@)r??9???1wɌ\8@:Preprocessing2F
Iterator::Model??{dsլ?!?a?=??E@)RD?U????1v???g3@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?~?Ϛ??!??IY??'@)\?M4???1????Z?@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip3?p?a???!??L@)?r߉y?1f?	?a@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?pY?? w?!K??u@)?pY?? w?1K??u@:Preprocessing2?
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[6]::Concatenate[1]::FromTensor?SH?9d?!?? ???)?SH?9d?1?? ???:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[6]::Concatenate[0]::TensorSlice?xy:W?b?!?2?M3??)?xy:W?b?1?2?M3??:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapׄ?Ơ??!?|??9?C@)?t><K?a?1?1?????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9pQ?B??`?I?z????X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	E??Ӝ???E??Ӝ???!E??Ӝ???      ??!       "      ??!       *      ??!       2	??(y$?@??(y$?@!??(y$?@:      ??!       B      ??!       J	?Hm?????Hm????!?Hm????R      ??!       Z	?Hm?????Hm????!?Hm????b      ??!       JCPU_ONLYYpQ?B??`?b q?z????X@