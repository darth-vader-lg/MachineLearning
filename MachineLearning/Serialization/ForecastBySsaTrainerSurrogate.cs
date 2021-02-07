using MachineLearning.Trainers;
using Microsoft.ML.Transforms.TimeSeries;
using System.Runtime.Serialization;

namespace MachineLearning.Serialization
{
   internal class ForecastBySsaTrainerSurrogate
   {
      internal class TrainerOptionsSurrogate : ISerializationSurrogate<ForecastBySsaTrainer.TrainerOptions>
      {
         public void GetObjectData(object obj, SerializationInfo info, StreamingContext context)
         {
            var data = (ForecastBySsaTrainer.TrainerOptions)obj;
            info.AddValue(nameof(data.OutputColumnName), data.OutputColumnName);
            info.AddValue(nameof(data.InputColumnName), data.InputColumnName);
            info.AddValue(nameof(data.WindowSize), data.WindowSize);
            info.AddValue(nameof(data.SeriesLength), data.SeriesLength);
            info.AddValue(nameof(data.TrainSize), data.TrainSize);
            info.AddValue(nameof(data.Horizon), data.Horizon);
            info.AddValue(nameof(data.IsAdaptive), data.IsAdaptive);
            info.AddValue(nameof(data.DiscountFactor), data.DiscountFactor);
            info.AddValue(nameof(data.RankSelectionMethod), (byte)data.RankSelectionMethod);
            info.AddValue(nameof(data.Rank), data.Rank);
            info.AddValue(nameof(data.MaxRank), data.MaxRank);
            info.AddValue(nameof(data.ShouldStabilize), data.ShouldStabilize);
            info.AddValue(nameof(data.ShouldMaintainInfo), data.ShouldMaintainInfo);
            info.AddValue(nameof(data.MaxGrowth), data.MaxGrowth);
            info.AddValue(nameof(data.ConfidenceLowerBoundColumn), data.ConfidenceLowerBoundColumn);
            info.AddValue(nameof(data.ConfidenceUpperBoundColumn), data.ConfidenceUpperBoundColumn);
            info.AddValue(nameof(data.ConfidenceLevel), data.ConfidenceLevel);
            info.AddValue(nameof(data.VariableHorizon), data.VariableHorizon);
         }
         public object SetObjectData(object obj, SerializationInfo info, StreamingContext context, ISurrogateSelector selector)
         {
            var data = new ForecastBySsaTrainer.TrainerOptions();
            info.Set(nameof(data.OutputColumnName), out data.OutputColumnName);
            info.Set(nameof(data.InputColumnName), out data.InputColumnName);
            info.Set(nameof(data.WindowSize), out data.WindowSize);
            info.Set(nameof(data.SeriesLength), out data.SeriesLength);
            info.Set(nameof(data.TrainSize), out data.TrainSize);
            info.Set(nameof(data.Horizon), out data.Horizon);
            info.Set(nameof(data.IsAdaptive), out data.IsAdaptive);
            info.Set(nameof(data.DiscountFactor), out data.DiscountFactor);
            data.RankSelectionMethod = (RankSelectionMethod)info.Set(nameof(data.RankSelectionMethod), out byte _);
            info.Set(nameof(data.Rank), out data.Rank);
            info.Set(nameof(data.MaxRank), out data.MaxRank);
            info.Set(nameof(data.ShouldStabilize), out data.ShouldStabilize);
            info.Set(nameof(data.ShouldMaintainInfo), out data.ShouldMaintainInfo);
            info.Set(nameof(data.MaxGrowth), out data.MaxGrowth);
            info.Set(nameof(data.ConfidenceLowerBoundColumn), out data.ConfidenceLowerBoundColumn);
            info.Set(nameof(data.ConfidenceUpperBoundColumn), out data.ConfidenceUpperBoundColumn);
            info.Set(nameof(data.ConfidenceLevel), out data.ConfidenceLevel);
            info.Set(nameof(data.VariableHorizon), out data.VariableHorizon);
            return data;
         }
      }
   }
}
