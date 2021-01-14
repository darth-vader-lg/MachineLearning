namespace MachineLearning.Model
{
   /// <summary>
   /// Interfaccia per i trainer di modello ciclici
   /// </summary>
   public interface IModelTrainerCycling : IModelTrainer
   {
      #region Properties
      /// <summary>
      /// Numero massimo di tentativi di training del modello
      /// </summary>
      int MaxTrainingCycles { get; }
      #endregion
   }
}
