using Microsoft.ML;
using System;
using TTransformer = Microsoft.ML.Trainers.FieldAwareFactorizationMachinePredictionTransformer;
using TTrainer = Microsoft.ML.Trainers.FieldAwareFactorizationMachineTrainer;
using TOptions = Microsoft.ML.Trainers.FieldAwareFactorizationMachineTrainer.Options;

namespace MachineLearning.Trainers
{
   /// <summary>
   /// Classe FieldAwareFactorizationMachineTrainer con opzioni
   /// </summary>
   [Serializable]
   public sealed class FieldAwareFactorizationMachineTrainer : TrainerBase<TTransformer, TTrainer, TOptions>
   {
      #region Methods
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="contextProvider">Provider di contesto di machine learning</param>
      internal FieldAwareFactorizationMachineTrainer(IContextProvider<MLContext> contextProvider, TOptions options = default) : base(contextProvider, options) { }
      /// <summary>
      /// Funzione di creazione del trainer
      /// </summary>
      /// <param name="context">Contesto di machine learning</param>
      /// <returns>Il trainer</returns>
      protected override TTrainer CreateTrainer(MLContext context) => context.BinaryClassification.Trainers.FieldAwareFactorizationMachine(Options);
      #endregion
   }
}
