using Microsoft.ML;
using System;
using TOptions = Microsoft.ML.Trainers.OlsTrainer.Options;
using TTrainer = Microsoft.ML.Trainers.OlsTrainer;
using TTransformer = Microsoft.ML.ISingleFeaturePredictionTransformer<Microsoft.ML.Trainers.OlsModelParameters>;

namespace MachineLearning.Trainers
{
   /// <summary>
   /// Classe SdcaRegressionTrainer con opzioni
   /// </summary>
   [Serializable]
   public sealed class OlsTrainer : TrainerBase<TTransformer, TTrainer, TOptions>
   {
      #region Methods
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="contextProvider">Provider di contesto di machine learning</param>
      internal OlsTrainer(IContextProvider<MLContext> contextProvider, TOptions options = default) : base(contextProvider, options) { }
      /// <summary>
      /// Funzione di creazione del trainer
      /// </summary>
      /// <param name="context">Contesto di machine learning</param>
      /// <returns>Il trainer</returns>
      protected override TTrainer CreateTrainer(MLContext context) => context.Regression.Trainers.Ols(Options);
      #endregion
   }
}
