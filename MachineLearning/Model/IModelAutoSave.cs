namespace MachineLearning.Model
{
   /// <summary>
   /// Interfaccia per i modelli con possibilita' di salvataggio automatico
   /// </summary>
   public interface IModelAutoSave
   {
      #region Properties
      /// <summary>
      /// Abilitazione al salvataggio automatico del modello
      /// </summary>
      bool ModelAutoSave { get; }
      #endregion
   }
}
