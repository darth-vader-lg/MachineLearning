namespace MachineLearning.Model
{
   /// <summary>
   /// Interfaccia per i provider di storage di modelli
   /// </summary>
   public interface IModelStorageProvider
   {
      #region Properties
      /// <summary>
      /// Storage del modello
      /// </summary>
      IModelStorage ModelStorage { get; }
      #endregion
   }
}
