using Microsoft.ML;
using System;
using TModel = Microsoft.ML.Trainers.MaximumEntropyModelParameters;
using TTransformer = Microsoft.ML.ISingleFeaturePredictionTransformer<Microsoft.ML.Trainers.MaximumEntropyModelParameters>;
using TTrainer = Microsoft.ML.Trainers.SdcaMaximumEntropyMulticlassTrainer;
using TOptions = Microsoft.ML.Trainers.SdcaMaximumEntropyMulticlassTrainer.Options;

namespace MachineLearning.Trainers
{
   /// <summary>
   /// Classe SdcaMaximumEntropyMulticlassTrainer con opzioni
   /// </summary>
   [Serializable]
   public sealed partial class SdcaMaximumEntropyMulticlassTrainer : TrainerBase<TModel, TTransformer, TTrainer, TOptions>
   {
      #region Methods
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="contextProvider">Provider di contesto</param>
      internal SdcaMaximumEntropyMulticlassTrainer(IContextProvider<MLContext> contextProvider, TOptions options = default) : base(contextProvider, options) { }
      /// <summary>
      /// Funzione di creazione del trainer
      /// </summary>
      /// <param name="context">Contesto di machine learning</param>
      /// <returns>Il trainer</returns>
      protected override TTrainer CreateTrainer(MLContext context) => context.MulticlassClassification.Trainers.SdcaMaximumEntropy(Options);
      #endregion
   }
}
