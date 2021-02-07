using Microsoft.ML;
using System;
using TOptions = Microsoft.ML.Trainers.LightGbm.LightGbmMulticlassTrainer.Options;
using TTrainer = Microsoft.ML.Trainers.LightGbm.LightGbmMulticlassTrainer;
using TTransformer = Microsoft.ML.Data.MulticlassPredictionTransformer<Microsoft.ML.Trainers.OneVersusAllModelParameters>;

namespace MachineLearning.Trainers
{
   /// <summary>
   /// Classe LightGbmMulticlassTrainer con opzioni
   /// </summary>
   [Serializable]
   public sealed partial class LightGbmMulticlassTrainer : TrainerBase<TTransformer, TTrainer, TOptions>
   {
      #region Methods
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="contextProvider">Provider di contesto</param>
      internal LightGbmMulticlassTrainer(IContextProvider<MLContext> contextProvider, TOptions options = default) : base(contextProvider, options) { }
      /// <summary>
      /// Funzione di creazione del trainer
      /// </summary>
      /// <param name="context">Contesto di machine learning</param>
      /// <returns>Il trainer</returns>
      protected override TTrainer CreateTrainer(MLContext context) => context.MulticlassClassification.Trainers.LightGbm(Options);
      #endregion
   }
}
