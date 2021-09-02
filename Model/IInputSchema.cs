using MachineLearning.Data;

namespace MachineLearning.Model
{
   /// <summary>
   /// Interfaccia per i modelli con definizione di schema di input
   /// </summary>
   public interface IInputSchema
   {
      #region Properties
      /// <summary>
      /// Schema di input del modello
      /// </summary>
      DataSchema InputSchema { get; }
      #endregion
   }
}
