using Microsoft.ML;
using System;
using TOptions = Microsoft.ML.Trainers.LbfgsMaximumEntropyMulticlassTrainer.Options;
using TTrainer = Microsoft.ML.Trainers.LbfgsMaximumEntropyMulticlassTrainer;
using TTransformer = Microsoft.ML.Data.MulticlassPredictionTransformer<Microsoft.ML.Trainers.MaximumEntropyModelParameters>;

namespace MachineLearning.Trainers
{
   /// <summary>
   /// Classe LbfgsMaximumEntropyMulticlassTrainer con opzioni
   /// </summary>
   [Serializable]
   public sealed partial class LbfgsMaximumEntropyMulticlassTrainer : TrainerBase<TTransformer, TTrainer, TOptions>
   {
      #region Methods
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="contextProvider">Provider di contesto</param>
      internal LbfgsMaximumEntropyMulticlassTrainer(IContextProvider<MLContext> contextProvider, TOptions options = default) : base(contextProvider, options) { }
      /// <summary>
      /// Funzione di creazione del trainer
      /// </summary>
      /// <param name="context">Contesto di machine learning</param>
      /// <returns>Il trainer</returns>
      protected override TTrainer CreateTrainer(MLContext context) => context.MulticlassClassification.Trainers.LbfgsMaximumEntropy(Options);
      #endregion
   }
}
