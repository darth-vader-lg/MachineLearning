using Microsoft.ML;
using Microsoft.ML.Transforms.TimeSeries;
using System;
using TOptions = MachineLearning.Trainers.ForecastBySsaTrainer.TrainerOptions;
using TTrainer = Microsoft.ML.Transforms.TimeSeries.SsaForecastingEstimator;
using TTransformer = Microsoft.ML.Transforms.TimeSeries.SsaForecastingTransformer;

namespace MachineLearning.Trainers
{
   /// <summary>
   /// Classe ForecastingBySsaTrainer con opzioni
   /// </summary>
   [Serializable]
   public sealed class ForecastBySsaTrainer : TrainerBase<TTransformer, TTrainer, TOptions>
   {
      #region Methods
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="contextProvider">Provider di contesto di machine learning</param>
      internal ForecastBySsaTrainer(IContextProvider<MLContext> contextProvider, TOptions options = default) : base(contextProvider, options) { }
      /// <summary>
      /// Funzione di creazione del trainer
      /// </summary>
      /// <param name="context">Contesto di machine learning</param>
      /// <returns>Il trainer</returns>
      protected override TTrainer CreateTrainer(MLContext context) =>
         context.Forecasting.ForecastBySsa(
            Options.OutputColumnName,
            Options.InputColumnName,
            Options.WindowSize,
            Options.SeriesLength,
            Options.TrainSize,
            Options.Horizon,
            Options.IsAdaptive,
            Options.DiscountFactor,
            Options.RankSelectionMethod,
            Options.Rank,
            Options.MaxRank,
            Options.ShouldStabilize,
            Options.ShouldMaintainInfo,
            Options.MaxGrowth,
            Options.ConfidenceLowerBoundColumn,
            Options.ConfidenceUpperBoundColumn,
            Options.ConfidenceLevel,
            Options.VariableHorizon);
      #endregion
      #region TrainerOptions
      /// <summary>
      /// Opzioni di training
      /// </summary>
      public sealed class TrainerOptions
      {
         /// <summary>
         /// Name of the column resulting from the transformation of inputColumnName.
         /// </summary>
         public string OutputColumnName;
         /// <summary>
         /// Name of column to transform. If set to null, the value of the outputColumnName will be used as source. The vector contains Alert, Raw Score, P-Value as first three values.
         /// </summary>
         public string InputColumnName;
         /// <summary>
         /// The length of the window on the series for building the trajectory matrix (parameter L).
         /// </summary>
         public int WindowSize;
         /// <summary>
         /// The length of series that is kept in buffer for modeling (parameter N).
         /// </summary>
         public int SeriesLength;
         /// <summary>
         /// The length of series from the beginning used for training.
         /// </summary>
         public int TrainSize;
         /// <summary>
         /// The number of values to forecast.
         /// </summary>
         public int Horizon;
         /// <summary>
         /// The flag determing whether the model is adaptive.
         /// </summary>
         public bool IsAdaptive;
         /// <summary>
         /// The discount factor in [0,1] used for online updates.
         /// </summary>
         public float DiscountFactor = 1;
         /// <summary>
         /// The rank selection method.
         /// </summary>
         public RankSelectionMethod RankSelectionMethod = RankSelectionMethod.Exact;
         /// <summary>
         /// The desired rank of the subspace used for SSA projection (parameter r). This parameter should be in the range in [1, windowSize].
         /// If set to null, the rank is automatically determined based on prediction error minimization.
         /// </summary>
         public int? Rank;
         /// <summary>
         /// The maximum rank considered during the rank selection process. If not provided (i.e. set to null), it is set to windowSize - 1.
         /// </summary>
         public int? MaxRank;
         /// <summary>
         /// The flag determining whether the model should be stabilized.
         /// </summary>
         public bool ShouldStabilize = true;
         /// <summary>
         /// The flag determining whether the meta information for the model needs to be maintained.
         /// </summary>
         public bool ShouldMaintainInfo;
         /// <summary>
         /// The maximum growth on the exponential trend.
         /// </summary>
         public GrowthRatio? MaxGrowth;
         /// <summary>
         /// The name of the confidence interval lower bound column. If not specified then confidence intervals will not be calculated.
         /// </summary>
         public string ConfidenceLowerBoundColumn;
         /// <summary>
         /// The name of the confidence interval upper bound column. If not specified then confidence intervals will not be calculated.
         /// </summary>
         public string ConfidenceUpperBoundColumn;
         /// <summary>
         /// The confidence level for forecasting.
         /// </summary>
         public float ConfidenceLevel = 0.95f;
         /// <summary>
         /// Set this to true if horizon will change after training(at prediction time).
         /// </summary>
         public bool VariableHorizon;
      }
      #endregion
   }
}
