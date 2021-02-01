using Microsoft.ML;
using Microsoft.ML.Trainers;
using System.Runtime.Serialization;

namespace MachineLearning.Serialization
{
   internal abstract class LbfgsTrainerBaseSurrogate<TOptions, TTransformer, TModel>
      where TOptions : LbfgsTrainerBase<TOptions, TTransformer, TModel>.OptionsBase, new()
      where TTransformer : ISingleFeaturePredictionTransformer<TModel>
      where TModel : class
   {
      internal abstract class OptionsBaseSurrogate : TrainerInputBaseWithWeightSurrogate
      {
         protected static new void GetObjectData(object obj, SerializationInfo info)
         {
            TrainerInputBaseWithWeightSurrogate.GetObjectData(obj, info);
            var data = (LbfgsTrainerBase<TOptions, TTransformer, TModel>.OptionsBase)obj;
            info.AddValue(nameof(data.L2Regularization), data.L2Regularization);
            info.AddValue(nameof(data.L1Regularization), data.L1Regularization);
            info.AddValue(nameof(data.OptimizationTolerance), data.OptimizationTolerance);
            info.AddValue(nameof(data.HistorySize), data.HistorySize);
            info.AddValue(nameof(data.MaximumNumberOfIterations), data.MaximumNumberOfIterations);
            info.AddValue(nameof(data.StochasticGradientDescentInitilaizationTolerance), data.StochasticGradientDescentInitilaizationTolerance);
            info.AddValue(nameof(data.Quiet), data.Quiet);
            info.AddValue(nameof(data.InitialWeightsDiameter), data.InitialWeightsDiameter);
            info.AddValue(nameof(data.NumberOfThreads), data.NumberOfThreads);
            info.AddValue(nameof(data.DenseOptimizer), data.DenseOptimizer);
            info.AddValue(nameof(data.EnforceNonNegativity), data.EnforceNonNegativity);
         }
         protected static new object SetObjectData(object obj, SerializationInfo info)
         {
            var data = (LbfgsTrainerBase<TOptions, TTransformer, TModel>.OptionsBase)TrainerInputBaseWithWeightSurrogate.SetObjectData(obj, info);
            info.Set(nameof(data.L2Regularization), out data.L2Regularization);
            info.Set(nameof(data.L1Regularization), out data.L1Regularization);
            info.Set(nameof(data.OptimizationTolerance), out data.OptimizationTolerance);
            info.Set(nameof(data.HistorySize), out data.HistorySize);
            info.Set(nameof(data.MaximumNumberOfIterations), out data.MaximumNumberOfIterations);
            info.Set(nameof(data.StochasticGradientDescentInitilaizationTolerance), out data.StochasticGradientDescentInitilaizationTolerance);
            info.Set(nameof(data.Quiet), out data.Quiet);
            info.Set(nameof(data.InitialWeightsDiameter), out data.InitialWeightsDiameter);
            info.Set(nameof(data.NumberOfThreads), out data.NumberOfThreads);
            info.Set(nameof(data.DenseOptimizer), out data.DenseOptimizer);
            info.Set(nameof(data.EnforceNonNegativity), out data.EnforceNonNegativity);
            return data;
         }
      }
   }
}
