using Microsoft.ML;
using System;
using TModel = Microsoft.ML.Trainers.FastTree.FastForestBinaryModelParameters;
using TTransformer = Microsoft.ML.Data.BinaryPredictionTransformer<Microsoft.ML.Trainers.FastTree.FastForestBinaryModelParameters>;
using TTrainer = Microsoft.ML.Trainers.FastTree.FastForestBinaryTrainer;
using TOptions = Microsoft.ML.Trainers.FastTree.FastForestBinaryTrainer.Options;

namespace MachineLearning.Trainers
{
   /// <summary>
   /// Classe FastForestBinaryTrainer con opzioni
   /// </summary>
   [Serializable]
   public sealed class FastForestBinaryTrainer : TrainerBase<TModel, TTransformer, TTrainer, TOptions>
   {
      #region Methods
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="contextProvider">Provider di contesto di machine learning</param>
      internal FastForestBinaryTrainer(IContextProvider<MLContext> contextProvider, TOptions options = default) : base(contextProvider, options) { }
      /// <summary>
      /// Funzione di creazione del trainer
      /// </summary>
      /// <param name="context">Contesto di machine learning</param>
      /// <returns>Il trainer</returns>
      protected override TTrainer CreateTrainer(MLContext context) => context.BinaryClassification.Trainers.FastForest(Options);
      #endregion
   }
}
