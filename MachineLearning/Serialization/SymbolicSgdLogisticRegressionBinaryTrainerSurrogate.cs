using Microsoft.ML.Trainers;
using System.Runtime.Serialization;

namespace MachineLearning.Serialization
{
   internal class SymbolicSgdLogisticRegressionBinaryTrainerSurrogate
   {
      internal class OptionsSurrogate : TrainerInputBaseWithLabelSurrogate, ISerializationSurrogate<SymbolicSgdLogisticRegressionBinaryTrainer.Options>
      {
         public void GetObjectData(object obj, SerializationInfo info, StreamingContext context)
         {
            GetObjectData(obj, info);
            var data = (SymbolicSgdLogisticRegressionBinaryTrainer.Options)obj;
            info.AddValue(nameof(data.NumberOfThreads), data.NumberOfThreads);
            info.AddValue(nameof(data.NumberOfIterations), data.NumberOfIterations);
            info.AddValue(nameof(data.Tolerance), data.Tolerance);
            info.AddValue(nameof(data.LearningRate), data.LearningRate);
            info.AddValue(nameof(data.L2Regularization), data.L2Regularization);
            info.AddValue(nameof(data.UpdateFrequency), data.UpdateFrequency);
            info.AddValue(nameof(data.MemorySize), data.MemorySize);
            info.AddValue(nameof(data.Shuffle), data.Shuffle);
            info.AddValue(nameof(data.PositiveInstanceWeight), data.PositiveInstanceWeight);
         }
         public object SetObjectData(object obj, SerializationInfo info, StreamingContext context, ISurrogateSelector selector)
         {
            var data = new SymbolicSgdLogisticRegressionBinaryTrainer.Options();
            SetObjectData(data, info);
            info.Set(nameof(data.NumberOfThreads), out data.NumberOfThreads);
            info.Set(nameof(data.NumberOfIterations), out data.NumberOfIterations);
            info.Set(nameof(data.Tolerance), out data.Tolerance);
            info.Set(nameof(data.LearningRate), out data.LearningRate);
            info.Set(nameof(data.L2Regularization), out data.L2Regularization);
            info.Set(nameof(data.UpdateFrequency), out data.UpdateFrequency);
            info.Set(nameof(data.MemorySize), out data.MemorySize);
            info.Set(nameof(data.Shuffle), out data.Shuffle);
            info.Set(nameof(data.PositiveInstanceWeight), out data.PositiveInstanceWeight);
            return data;
         }
      }
   }
}
