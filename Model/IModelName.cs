namespace MachineLearning.Model
{
   /// <summary>
   /// Interfaccia per i modelli con nome
   /// </summary>
   public interface IModelName
   {
      #region Properties
      /// <summary>
      /// Nome del modello
      /// </summary>
      string ModelName { get; }
      #endregion
   }
}
