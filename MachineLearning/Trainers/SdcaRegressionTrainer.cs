using Microsoft.ML;
using System;
using TModel = Microsoft.ML.Trainers.LinearMulticlassModelParameters;
using TTransformer = Microsoft.ML.ISingleFeaturePredictionTransformer<Microsoft.ML.Trainers.LinearMulticlassModelParameters>;
using TTrainer = Microsoft.ML.Trainers.SdcaRegressionTrainer;
using TOptions = Microsoft.ML.Trainers.SdcaRegressionTrainer.Options;

namespace MachineLearning.Trainers
{
   /// <summary>
   /// Classe SdcaRegressionTrainer con opzioni
   /// </summary>
   [Serializable]
   public sealed class SdcaRegressionTrainer : TrainerBase<TModel, TTransformer, TTrainer, TOptions>
   {
      #region Methods
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="contextProvider">Provider di contesto di machine learning</param>
      internal SdcaRegressionTrainer(IContextProvider<MLContext> contextProvider, TOptions options = default) : base(contextProvider, options) { }
      /// <summary>
      /// Funzione di creazione del trainer
      /// </summary>
      /// <param name="context">Contesto di machine learning</param>
      /// <returns>Il trainer</returns>
      protected override TTrainer CreateTrainer(MLContext context) => context.Regression.Trainers.Sdca(Options);
      #endregion
   }
}
