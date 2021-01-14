namespace MachineLearning.Model
{
   /// <summary>
   /// Interfaccia per i provider di training del modello
   /// </summary>
   public interface IModelTrainerProvider
   {
      #region Properties
      /// <summary>
      /// Trainer del modello
      /// </summary>
      IModelTrainer ModelTrainer { get; }
      #endregion
   }
}
