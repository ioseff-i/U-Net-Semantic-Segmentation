	j??{??@j??{??@!j??{??@	{?~?d?{?~?d?!{?~?d?"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$j??{??@?j????A?f???@Y?C?3???*	/?$?O@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatū?m???!r????4B@)??~??Γ?1?ܣ>@:Preprocessing2U
Iterator::Model::ParallelMapV2q:Ɇ?!??>W??1@)q:Ɇ?1??>W??1@:Preprocessing2F
Iterator::Model??%????!?T?8	A@)??6?4D??1??c?r0@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?'?.????!t?x??U6@)t??Y5??1???:??*@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceu ???Ww?!??"@)u ???Ww?1??"@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??gy?m?!??!@)??gy?m?1??!@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipEׅ?O??!?U??u{P@)???3.l?1????@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?m?(??!?T?*8@)?/J?_?Q?1????C???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9{?~?d?I?????X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?j?????j????!?j????      ??!       "      ??!       *      ??!       2	?f???@?f???@!?f???@:      ??!       B      ??!       J	?C?3????C?3???!?C?3???R      ??!       Z	?C?3????C?3???!?C?3???b      ??!       JCPU_ONLYY{?~?d?b q?????X@