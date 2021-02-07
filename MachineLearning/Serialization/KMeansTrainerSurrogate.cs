using Microsoft.ML.Trainers;
using System.Runtime.Serialization;

namespace MachineLearning.Serialization
{
   internal class KMeansTrainerSurrogate
   {
      internal class OptionsSurrogate : UnsupervisedTrainerInputBaseWithWeightSurrogate, ISerializationSurrogate<KMeansTrainer.Options>
      {
         public void GetObjectData(object obj, SerializationInfo info, StreamingContext context)
         {
            GetObjectData(obj, info);
            var data = (KMeansTrainer.Options)obj;
            info.AddValue(nameof(data.NumberOfClusters), data.NumberOfClusters);
            info.AddValue(nameof(data.InitializationAlgorithm), (byte)data.InitializationAlgorithm);
            info.AddValue(nameof(data.OptimizationTolerance), data.OptimizationTolerance);
            info.AddValue(nameof(data.MaximumNumberOfIterations), data.MaximumNumberOfIterations);
            info.AddValue(nameof(data.AccelerationMemoryBudgetMb), data.AccelerationMemoryBudgetMb);
            info.AddValue(nameof(data.NumberOfThreads), data.NumberOfThreads);
         }
         public object SetObjectData(object obj, SerializationInfo info, StreamingContext context, ISurrogateSelector selector)
         {
            var data = new KMeansTrainer.Options();
            SetObjectData(data, info);
            info.Set(nameof(data.NumberOfClusters), out data.NumberOfClusters);
            data.InitializationAlgorithm = (KMeansTrainer.InitializationAlgorithm)info.Set(nameof(data.InitializationAlgorithm), out byte _);
            info.Set(nameof(data.OptimizationTolerance), out data.OptimizationTolerance);
            info.Set(nameof(data.MaximumNumberOfIterations), out data.MaximumNumberOfIterations);
            info.Set(nameof(data.AccelerationMemoryBudgetMb), out data.AccelerationMemoryBudgetMb);
            info.Set(nameof(data.NumberOfThreads), out data.NumberOfThreads);
            return data;
         }
      }
   }
}
